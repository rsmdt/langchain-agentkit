# ruff: noqa: N805
"""Real-LLM eval: skill progressive disclosure resolves the base-directory header.

Exercises the full contract end-to-end:

1. A skill is discovered from a backend, so its body is prefixed with a
   ``Base directory for this "<name>" skill: <dir>`` header.
2. The agent loads the skill via the ``Skill`` tool and sees that header.
3. The skill body points at ``references/procedure.md`` *relative to the base
   directory*, so the agent must resolve the announced anchor and call ``Read``
   with the correct absolute (backend-relative) path — immediately, without
   searching with Glob/Grep first.
4. Because the ``Skill`` and ``Read`` tools share one backend, the announced
   path is directly readable and the reference content reaches the model.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_skill_reference_eval.py -x -v -m eval
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_agentkit import Agent, FilesystemExtension
from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.extensions.skills import SkillsExtension
from tests.evals.conftest import EVAL_MODEL
from tests.evals.eval_runner import extract_tool_calls_from_messages

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment,misc]


_SKILL_MD = """\
---
name: data-export
description: Export account data to CSV using the team's authoritative, versioned export procedure.
---
# Data Export

Do NOT answer from memory — the procedure changes often.

The authoritative, current export procedure lives in `references/procedure.md`,
relative to this skill's base directory shown above. You MUST read that file with
the Read tool before responding (resolve the path against the base directory),
then follow it exactly.
"""

# A unique marker the model could not know without actually reading the file.
_PROCEDURE_MARKER = "EXPORT-7731"
_PROCEDURE_MD = f"""\
# Export Procedure (v7)

1. Open Settings -> Data.
2. Click "Export to CSV".
3. Confirm with passphrase {_PROCEDURE_MARKER}.
"""


def _get_llm():
    """Return a deterministic ChatOpenAI instance."""
    return ChatOpenAI(model=EVAL_MODEL, temperature=0)


def _write_skill_with_reference(root: Path) -> None:
    """Create a skill directory with a reference file under *root*."""
    skill_dir = root / "data-export"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD)
    (skill_dir / "references" / "procedure.md").write_text(_PROCEDURE_MD)


async def _build_agent(skills_ext: SkillsExtension, fs_ext: FilesystemExtension):
    """Build a ReAct agent wired with the skills and filesystem extensions."""
    _llm = _get_llm()

    class Worker(Agent):
        model = _llm
        extensions = [skills_ext, fs_ext]
        prompt = (
            "You complete user requests using available skills. When a skill matches "
            "the request, load it with the Skill tool and follow its instructions "
            "exactly — including reading any reference files it points to, at the "
            "location it specifies. Do not search for files you have already been "
            "given a path to."
        )

        async def handler(state, *, llm, tools, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.bind_tools(tools).ainvoke(messages)
            return {"messages": [response], "sender": "worker"}

    return (await Worker().graph()).compile()


class TestSkillReferenceResolution:
    """The agent resolves the base-directory header to read a skill's reference."""

    async def test_reads_reference_at_announced_location_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill_with_reference(Path(tmp))

            # One shared backend: the header the skill announces is therefore a
            # path the Read tool can resolve.
            backend = OSBackend(tmp)
            skills_ext = SkillsExtension(skills="/", backend=backend)
            await skills_ext.setup()
            fs_ext = FilesystemExtension(backend=backend)

            # The base directory the skill body actually announces, parsed from
            # the loaded config's header — the eval asserts the model used *this*.
            header_line = skills_ext.configs[0].prompt.splitlines()[0]
            base_dir = header_line.split(":", 1)[1].strip()
            expected_path = f"{base_dir}/references/procedure.md"

            agent = await _build_agent(skills_ext, fs_ext)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="How do I export my account data to CSV?")]}
            )
            messages = result["messages"]
            calls = extract_tool_calls_from_messages(messages)

            # 1. The skill was loaded.
            skill_idx = next(
                (
                    i
                    for i, c in enumerate(calls)
                    if c["name"] == "Skill" and c["args"].get("skill_name") == "data-export"
                ),
                None,
            )
            assert skill_idx is not None, f"Skill(data-export) was never loaded. Calls: {calls}"

            # 2. The first file access after loading the skill is a Read — the
            #    agent went straight to the file, no Glob/Grep search.
            after = [c for c in calls[skill_idx + 1 :] if c["name"] in {"Read", "Glob", "Grep"}]
            assert after, f"No file access after loading the skill. Calls: {calls}"
            first = after[0]
            assert first["name"] == "Read", (
                f"Expected Read immediately after loading the skill, got {first['name']}. "
                f"Calls: {calls}"
            )

            # 3. That Read targets the location the header announced.
            assert first["args"].get("file_path") == expected_path, (
                f"Read used the wrong location. Expected {expected_path!r}, "
                f"got {first['args'].get('file_path')!r}. Calls: {calls}"
            )

            # 4. The shared backend made the path readable — the reference content
            #    actually reached the model.
            assert any(
                isinstance(m, ToolMessage) and _PROCEDURE_MARKER in str(m.content) for m in messages
            ), f"Reference content ({_PROCEDURE_MARKER}) never reached the model"
