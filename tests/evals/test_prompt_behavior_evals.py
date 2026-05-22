"""LLM behavioral evals for AgentKit's current prompts and tool descriptions.

Each eval constructs an ``AgentKit`` the same way the ``Agent`` class does
(``run_extension_setup`` then either ``kit.compile(handler)`` or a direct
model invocation), sends one user message, and asserts on the resulting
tool calls or assistant text. The purpose is to confirm the LLM behaves
correctly given the prompts that ship today.

Run::

    pytest tests/evals/test_prompt_behavior_evals.py -v -m eval

Skipped cleanly when ``OPENAI_API_KEY`` is unset.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.evals.conftest import EVAL_MODEL

pytestmark = pytest.mark.eval

try:
    from langchain_openai import ChatOpenAI

    _HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
except ImportError:  # pragma: no cover — import guard
    _HAS_OPENAI = False

SKIP_REASON = "Requires OPENAI_API_KEY and langchain-openai"
MODEL_NAME = EVAL_MODEL

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm() -> Any:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _make_kit(extensions: list[Any], **kw: Any) -> Any:
    """Build + setup an AgentKit with the given extensions."""
    from langchain_agentkit import AgentKit
    from langchain_agentkit.agent_kit import run_extension_setup

    kit = AgentKit(extensions=extensions, **kw)
    asyncio.run(run_extension_setup(kit))
    return kit


async def _invoke_once(
    kit: Any,
    user_text: str,
    *,
    extra_state: dict[str, Any] | None = None,
    tool_choice: Any = None,
) -> AIMessage:
    """Run one LLM turn the way the compiled graph does.

    Builds the system message from ``kit.compose(state).prompt``. The
    reminder channel has been removed — all dynamic content now lives
    in the system prompt, re-rendered per step.
    """
    state: dict[str, Any] = {"messages": [HumanMessage(content=user_text)]}
    if extra_state:
        state.update(extra_state)

    composition = kit.compose(state)
    messages: list[Any] = []
    if composition.prompt:
        messages.append(SystemMessage(content=composition.prompt))
    messages.extend(state["messages"])

    bind_kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        bind_kwargs["tool_choice"] = tool_choice
    llm = _llm()
    bound = llm.bind_tools(kit.tools, **bind_kwargs) if kit.tools else llm
    response = await bound.ainvoke(messages)
    assert isinstance(response, AIMessage)
    return response


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _tool_names(msg: AIMessage) -> list[str]:
    return [tc["name"] for tc in (msg.tool_calls or [])]


def _best_of_three(fn: Any, *, min_pass: int = 2) -> tuple[int, list[str]]:
    """Run ``fn()`` 3 times; return (passes, failure_comments)."""
    passes = 0
    comments: list[str] = []
    for i in range(3):
        try:
            ok, comment = fn()
        except Exception as exc:  # pragma: no cover — surface as comment
            ok, comment = False, f"trial {i}: raised {exc!r}"
        if ok:
            passes += 1
        else:
            comments.append(f"trial {i}: {comment}")
    return passes, comments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repo_fixture() -> Path:
    """Create a tiny repo-like tree for filesystem evals."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_pp_repo_"))
    (tmp / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.0.1"\n',
    )
    (tmp / "README.md").write_text("# Demo\n\nA demo project.\n")
    (tmp / "src").mkdir()
    (tmp / "src" / "app.py").write_text("# TODO: implement\nprint('hi')\n")
    (tmp / "tests").mkdir()
    (tmp / "tests" / "test_app.py").write_text("def test_ok():\n    assert True\n")
    (tmp / "tests" / "test_other.py").write_text("def test_other():\n    assert 1 == 1\n")
    return tmp


@pytest.fixture(scope="module")
def memory_fixture() -> Path:
    """Create a MEMORY.md directory layout for MemoryExtension."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_pp_mem_"))
    (tmp / "MEMORY.md").write_text(
        "# User Preferences\n\n"
        "The user prefers metric units (kilometers, meters, celsius). "
        "Never convert distances to miles or feet — keep them in metric. "
        "When the user asks about a distance in 'familiar' or 'everyday' "
        "units, metric *is* the familiar unit for this user.\n",
    )
    return tmp


_GIT_WORDS = re.compile(r"\b(git|commit|gh|pr)\b", re.IGNORECASE)


def _contains_git_language(msg: AIMessage) -> str | None:
    """Return a snippet if the message has any git/PR language, else None."""
    text_parts: list[str] = []
    if isinstance(msg.content, str):
        text_parts.append(msg.content)
    elif isinstance(msg.content, list):
        for part in msg.content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
    for tc in msg.tool_calls or []:
        args = tc.get("args") or {}
        for v in args.values():
            if isinstance(v, str):
                text_parts.append(v)
    for text in text_parts:
        m = _GIT_WORDS.search(text)
        if m:
            return text[max(0, m.start() - 20) : m.end() + 20]
    return None


# ---------------------------------------------------------------------------
# Tool-description behavior (Bash / Read / Grep / Glob / Agent / WebSearch / AskUser)
# ---------------------------------------------------------------------------


def _fs_kit(root: Path) -> Any:
    """A real FilesystemExtension kit.

    Composes the extension itself — NOT a hand-picked tool list — so the
    eval exercises the full shipped surface: every FS tool, Bash, AND the
    extension's system-prompt guidance (e.g. the dedicated-tool-over-Bash
    appendix from FilesystemExtension.prompt()). Hand-wiring tools with
    extensions=[] skips that guidance and silently under-tests the extension.
    """
    from langchain_agentkit import FilesystemExtension

    return _make_kit(extensions=[FilesystemExtension(root=str(root))])


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestBashNoGitProtocol:
    def test_no_git_language(self, repo_fixture: Path) -> None:
        prompts = [
            "Show me what files are in the current directory.",
            "Clean up this folder.",
            "What changed recently?",
        ]
        failures: list[str] = []
        for prompt in prompts:
            kit = _fs_kit(repo_fixture)

            def trial(kit: Any = kit, prompt: str = prompt) -> tuple[bool, str]:
                msg = _run(_invoke_once(kit, prompt))
                leak = _contains_git_language(msg)
                if leak is not None:
                    return False, f"git language leaked: {leak!r}"
                return True, ""

            passes, comments = _best_of_three(trial)
            if passes < 2:
                failures.append(f"{prompt!r}: {passes}/3 passed; {comments}")
        assert not failures, "\n".join(failures)


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestBashRetainsShellEssentials:
    def test_bash_invoked_for_listing_python_files(self, repo_fixture: Path) -> None:
        def trial() -> tuple[bool, str]:
            kit = _fs_kit(repo_fixture)
            msg = _run(
                _invoke_once(
                    kit,
                    "Run a shell one-liner that counts the total number of "
                    "lines across all .py files in this repo.",
                )
            )
            names = _tool_names(msg)
            if "Bash" not in names:
                return False, f"no Bash call; tool_calls={names}, content={msg.content!r}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestReadPreferredOverBash:
    def test_prefers_read(self, repo_fixture: Path) -> None:
        target = repo_fixture / "pyproject.toml"

        def trial() -> tuple[bool, str]:
            kit = _fs_kit(repo_fixture)
            msg = _run(_invoke_once(kit, f"What's in the pyproject.toml file at {target}?"))
            names = _tool_names(msg)
            if "Read" in names and "Bash" not in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestGrepPreferredOverBash:
    def test_prefers_grep(self, repo_fixture: Path) -> None:
        def trial() -> tuple[bool, str]:
            kit = _fs_kit(repo_fixture)
            msg = _run(_invoke_once(kit, "Find all occurrences of TODO in this directory."))
            names = _tool_names(msg)
            if "Grep" in names and "Bash" not in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestGlobPreferredOverBash:
    def test_prefers_glob(self, repo_fixture: Path) -> None:
        def trial() -> tuple[bool, str]:
            kit = _fs_kit(repo_fixture)
            msg = _run(_invoke_once(kit, "List all Python test files in this repo."))
            names = _tool_names(msg)
            if "Glob" in names and "Bash" not in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestWritePreferredOverBash:
    def test_prefers_write(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.txt"

        def trial() -> tuple[bool, str]:
            kit = _fs_kit(tmp_path)
            msg = _run(
                _invoke_once(
                    kit,
                    f"Create a file at {target} containing the text `Hello World`.",
                )
            )
            names = _tool_names(msg)
            if "Bash" in names:
                return False, f"Bash used; tool_calls={names}"
            for tc in msg.tool_calls or []:
                if tc["name"] != "Write":
                    continue
                args = tc.get("args") or {}
                if str(args.get("file_path", "")) == str(target) and "Hello World" in str(
                    args.get("content", "")
                ):
                    return True, ""
            return False, f"Write not called correctly; tool_calls={msg.tool_calls!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestEditPreferredOverBash:
    def test_prefers_edit(self, tmp_path: Path) -> None:
        target = tmp_path / "config.yaml"
        target.write_text("debug: true\nlog_level: info\n")

        def trial() -> tuple[bool, str]:
            kit = _fs_kit(tmp_path)
            msg = _run(
                _invoke_once(
                    kit,
                    f"In {target}, change `debug: true` to `debug: false`.",
                )
            )
            names = _tool_names(msg)
            if "Bash" in names:
                return False, f"Bash used instead of a dedicated FS tool; tool_calls={names}"
            # With the full FS toolset (the real extension), a well-behaved
            # model Reads the file before Editing — Edit's contract requires a
            # prior Read. A single turn therefore legitimately surfaces Read
            # first; either Read or Edit confirms the model chose dedicated FS
            # tools over Bash/sed for the edit.
            if "Edit" in names or "Read" in names:
                return True, ""
            return False, f"no dedicated FS tool used; tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestAgentDelegationParallel:
    def test_parallel_agent_calls_for_three_docs(self) -> None:
        from langchain_agentkit import AgentsExtension
        from langchain_agentkit.extensions.agents import AgentConfig

        researcher = AgentConfig(
            name="researcher",
            description="Researches and summarizes a single document or topic.",
            prompt="You are a researcher. Summarize the requested input concisely.",
            model=MODEL_NAME,
        )

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    AgentsExtension(
                        agents=[researcher],
                        default_conciseness=True,
                    )
                ],
                model=MODEL_NAME,
                model_resolver=lambda name: ChatOpenAI(model=name, temperature=0),
            )
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "I have three documents to summarize in parallel:\n"
                        "A) a Q3 vendor analysis report\n"
                        "B) a customer interview transcript\n"
                        "C) a competitive landscape overview\n"
                        "Delegate the summarization of each document to the researcher "
                        "subagent. Run them in parallel."
                    ),
                )
            )
            agent_calls = [n for n in _tool_names(msg) if n == "Agent"]
            if len(agent_calls) >= 2:
                return True, ""
            return False, f"tool_calls={_tool_names(msg)}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestWebSearchInvokedWithNoPromptFallback:
    def test_websearch_used(self) -> None:
        from langchain_agentkit import WebSearchExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[WebSearchExtension()])
            msg = _run(_invoke_once(kit, "What's the weather in Tokyo right now?"))
            names = _tool_names(msg)
            if "WebSearch" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestAskUserInvokedOnAmbiguity:
    def test_ask_user_on_ambiguous_prompt(self) -> None:
        from langchain_agentkit import HITLExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[HITLExtension()])
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "I'd like you to process one of my files. I have three: "
                        "report.csv, notes.txt, and summary.md. Pick the right one and "
                        "process it. Rules: do NOT answer in plain text, do NOT guess, "
                        "do NOT ask me to upload anything — you MUST resolve the "
                        "ambiguity by calling the AskUser tool with the three files "
                        "as options."
                    ),
                )
            )
            names = _tool_names(msg)
            if "AskUser" in names:
                return True, ""
            return False, f"tool_calls={names}, content={msg.content!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# CoreBehaviorExtension
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestCoreBehaviorTerseReplies:
    def test_short_answer_for_trivial_math(self) -> None:
        from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[CoreBehaviorExtension()])
            msg = _run(_invoke_once(kit, "What is 2+2?"))
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(text.strip()) <= 50 and "4" in text:
                return True, ""
            return False, f"len={len(text)}, content={text!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestCoreBehaviorParallelToolCalls:
    def test_multiple_tool_calls_in_one_turn(self, repo_fixture: Path) -> None:
        from langchain_agentkit import FilesystemExtension
        from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    CoreBehaviorExtension(),
                    FilesystemExtension(root=str(repo_fixture)),
                ],
            )
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "I need three things done in one step: (a) find all TODO comments "
                        "in this repo, (b) list every Python file in this repo, and "
                        f"(c) read the README.md at {repo_fixture / 'README.md'}."
                    ),
                )
            )
            if len(msg.tool_calls or []) >= 2:
                return True, ""
            return False, f"tool_calls={_tool_names(msg)}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestCoreBehaviorPreferDedicatedTools:
    def test_grep_chosen_for_class_search(self, repo_fixture: Path) -> None:
        from langchain_agentkit import FilesystemExtension
        from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    CoreBehaviorExtension(),
                    FilesystemExtension(root=str(repo_fixture)),
                ],
            )
            msg = _run(_invoke_once(kit, "Search for all usages of class_name `Foo`."))
            names = _tool_names(msg)
            if "Grep" in names and "Bash" not in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# Task workflow (TasksExtension + TeamExtension)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskCreateOnMultiStepRequest:
    def test_task_create_called(self) -> None:
        from langchain_agentkit import TasksExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[TasksExtension()])
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "Set up a new Python project: create pyproject.toml, write a "
                        "main module, and add a test."
                    ),
                )
            )
            if "TaskCreate" in _tool_names(msg):
                return True, ""
            return False, f"tool_calls={_tool_names(msg)}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskUpdateOnCompletion:
    def test_task_update_on_completed_step(self) -> None:
        from langchain_agentkit import TasksExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[TasksExtension()])
            tasks = [
                {
                    "id": "1",
                    "subject": "Create pyproject.toml",
                    "description": "Set up project metadata.",
                    "status": "in_progress",
                    "active_form": "Creating pyproject.toml",
                },
                {
                    "id": "2",
                    "subject": "Write main module",
                    "description": "Create the main module file.",
                    "status": "pending",
                },
            ]
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "I've finished creating pyproject.toml. Please update task #1 "
                        "to completed and move on."
                    ),
                    extra_state={"tasks": tasks},
                )
            )
            for tc in msg.tool_calls or []:
                if tc["name"] != "TaskUpdate":
                    continue
                args = tc.get("args") or {}
                if args.get("status") == "completed":
                    return True, ""
            return False, f"tool_calls={_tool_names(msg)}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskCreateTeamVariant:
    @pytest.mark.xfail(
        reason=(
            "Baseline finding (gpt-5.4-mini): with a TeamExtension sibling, "
            "TaskCreate's team-aware description and TeamCreate both claim the "
            "'assign work to a teammate' space, so the model picks TeamCreate for "
            "'create a task and assign it to the db-specialist'. Real tool-instruction "
            "conflict — tracked for the refactor to disambiguate the "
            "TaskCreate/TeamCreate boundary (docs/refactor/tool-descriptions.md)."
        ),
        strict=False,
    )
    def test_team_active_description_includes_owner(self) -> None:
        from langchain_agentkit import TasksExtension, TeamExtension
        from langchain_agentkit.extensions.agents import AgentConfig
        from langchain_agentkit.extensions.agents.types import _AgentConfigProxy

        specialist = _AgentConfigProxy(
            AgentConfig(
                name="db-specialist",
                description="Database specialist; investigates DB issues.",
                prompt="You investigate DB errors.",
            )
        )

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    TasksExtension(),
                    TeamExtension(agents=[specialist]),
                ]
            )
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "Create a task to investigate the database error and assign it "
                        "to the db-specialist teammate."
                    ),
                )
            )
            # The team-active TaskCreate description instructs the model
            # to (a) create the task, then (b) assign it via TaskUpdate
            # with an ``owner`` parameter.  Accept either signal: the
            # TaskCreate itself tagging a teammate in its description/
            # subject, a follow-up TaskUpdate carrying an ``owner``, or
            # an owner arg set on TaskCreate (some models will do this
            # despite the description).
            saw_create = False
            for tc in msg.tool_calls or []:
                name = tc["name"]
                args = tc.get("args") or {}
                if name == "TaskCreate":
                    saw_create = True
                    if args.get("owner"):
                        return True, ""
                    meta = args.get("metadata") or {}
                    if isinstance(meta, dict) and meta.get("owner"):
                        return True, ""
                    text = f"{args.get('subject', '')} {args.get('description', '')}"
                    if "db-specialist" in text or "specialist" in text.lower():
                        return True, ""
                elif name == "TaskUpdate" and args.get("owner"):
                    return True, ""
            if saw_create:
                return True, "TaskCreate called (team-aware description took effect)"
            return False, f"tool_calls={msg.tool_calls!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# Skill roster delivered in the system prompt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestSkillInvokedFromSystemPromptRoster:
    def test_skill_tool_invoked(self, skills_extension: Any) -> None:
        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[skills_extension])
            # The skills fixture exposes a "market-sizing" skill; the user
            # request matches its description. The roster is appended to
            # the system prompt by SkillsExtension.prompt().
            msg = _run(
                _invoke_once(
                    kit,
                    (
                        "Please use /market-sizing to estimate the TAM, SAM, and SOM "
                        "for an electric bike startup in Europe."
                    ),
                )
            )
            for tc in msg.tool_calls or []:
                if tc["name"] != "Skill":
                    continue
                args = tc.get("args") or {}
                # Tool input field name is implementation-dependent; accept
                # any arg value matching the fixture skill.
                values = [str(v).lower() for v in args.values()]
                if any("market-sizing" in v for v in values):
                    return True, ""
            return False, f"tool_calls={msg.tool_calls!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestSkillDisambiguation:
    def test_picks_market_sizing_skill(self, skills_extension: Any) -> None:
        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[skills_extension])
            msg = _run(
                _invoke_once(
                    kit,
                    "Help me estimate the TAM for cloud storage in Europe.",
                )
            )
            for tc in msg.tool_calls or []:
                if tc["name"] != "Skill":
                    continue
                args = tc.get("args") or {}
                values = [str(v).lower() for v in args.values()]
                if any("market-sizing" in v for v in values):
                    return True, ""
            return False, f"tool_calls={msg.tool_calls!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_picks_translate_skill(self, skills_extension: Any) -> None:
        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[skills_extension])
            msg = _run(
                _invoke_once(
                    kit,
                    "Please translate the sentence 'Good morning, how are you?' "
                    "from English into French.",
                )
            )
            for tc in msg.tool_calls or []:
                if tc["name"] != "Skill":
                    continue
                args = tc.get("args") or {}
                values = [str(v).lower() for v in args.values()]
                if any("translate" in v for v in values):
                    return True, ""
            return False, f"tool_calls={msg.tool_calls!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# MemoryExtension
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestMemoryContentInfluencesResponse:
    def test_metric_preference_reflected(self, memory_fixture: Path) -> None:
        from langchain_agentkit.extensions.memory import MemoryExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    MemoryExtension(path=memory_fixture, project_discovery=False),
                ],
            )
            msg = _run(_invoke_once(kit, "What's 100km in a familiar unit?"))
            text = (
                msg.content
                if isinstance(msg.content, str)
                else " ".join(str(p.get("text", "")) for p in msg.content if isinstance(p, dict))
            )
            lower = text.lower()
            uses_metric = "km" in lower or "kilomet" in lower or "meter" in lower
            converts_to_miles = "mile" in lower
            if uses_metric and not converts_to_miles:
                return True, ""
            return False, f"content={text!r}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.fixture()
def irrelevant_memory_fixture(tmp_path: Path) -> Path:
    """MEMORY.md containing content unrelated to the prompts under test."""
    (tmp_path / "MEMORY.md").write_text(
        "# User Preferences\n\nThe user's favorite color is blue.\n",
    )
    return tmp_path


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestMemoryIrrelevantContentDoesNotInterfere:
    def test_irrelevant_memory_does_not_leak(self, irrelevant_memory_fixture: Path) -> None:
        from langchain_agentkit.extensions.memory import MemoryExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(
                extensions=[
                    MemoryExtension(path=irrelevant_memory_fixture, project_discovery=False),
                ],
            )
            msg = _run(_invoke_once(kit, "What is 2 + 2?"))
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            lower = text.lower()
            if "4" not in text:
                return False, f"no '4' in content={text!r}"
            if "blue" in lower or "color" in lower or "colour" in lower:
                return False, f"irrelevant memory leaked; content={text!r}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestAskUserNotInvokedWhenClear:
    def test_no_ask_user_for_explicit_path(self, tmp_path: Path) -> None:
        from langchain_agentkit import HITLExtension

        target = tmp_path / "test.txt"
        target.write_text("hello\n")

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[HITLExtension()])
            msg = _run(_invoke_once(kit, f"Read the file at {target}."))
            names = _tool_names(msg)
            if "AskUser" in names:
                return False, f"AskUser invoked; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_no_ask_user_for_general_knowledge(self) -> None:
        from langchain_agentkit import HITLExtension

        def trial() -> tuple[bool, str]:
            kit = _make_kit(extensions=[HITLExtension()])
            msg = _run(_invoke_once(kit, "What is the capital of France?"))
            names = _tool_names(msg)
            if "AskUser" in names:
                return False, f"AskUser invoked; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# preset="full" smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestPresetFullSmoke:
    def test_preset_full_compiles_and_responds(self) -> None:
        kit = _make_kit(extensions=[], preset="full", model=_llm())

        # The preset must seed CoreBehavior + Tasks + Memory, so the
        # prompt channel is non-empty.
        composition = kit.compose({"messages": []})
        assert composition.prompt, "preset='full' should seed a non-empty prompt"

        msg = _run(_invoke_once(kit, "Give me a one-sentence greeting."))
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        assert text.strip(), "preset='full' agent returned empty text"


# ---------------------------------------------------------------------------
# Tool-selection coverage (gap closure)
#   See docs/refactor/tool-description-evals.md.
#
# Description-sensitive: each prompt expresses *intent* WITHOUT naming the tool,
# so a regression in a tool's description surfaces as a selection miss. Each
# tool gets a positive (right tool fires) and a negative (a plausible-but-wrong
# prompt does NOT fire it). These guard the description rewrites in
# docs/refactor/tool-descriptions.md.
# ---------------------------------------------------------------------------


_SEED_TASKS = [
    {
        "id": "1",
        "subject": "Set up database schema",
        "description": "Create tables and indexes for the orders domain.",
        "status": "pending",
    },
    {
        "id": "2",
        "subject": "Write API endpoints",
        "description": "Implement CRUD endpoints for orders. Depends on the schema.",
        "status": "in_progress",
        "active_form": "Writing API endpoints",
        "owner": "self",
    },
    {
        "id": "3",
        "subject": "Write integration tests",
        "description": "End-to-end tests for the orders flow. Depends on the API.",
        "status": "pending",
    },
]


def _tasks_kit() -> Any:
    from langchain_agentkit import TasksExtension

    return _make_kit(extensions=[TasksExtension()])


def _team_kit() -> Any:
    from langchain_agentkit import TeamExtension
    from langchain_agentkit.extensions.agents import AgentConfig
    from langchain_agentkit.extensions.agents.types import _AgentConfigProxy

    researcher = _AgentConfigProxy(
        AgentConfig(
            name="researcher",
            description="Researches topics and summarizes findings.",
            prompt="You research a topic and summarize findings concisely.",
        )
    )
    return _make_kit(extensions=[TeamExtension(agents=[researcher])])


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskGetSelection:
    def test_selects_task_get_for_single_task_details(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The task list only shows titles. Fetch the complete stored "
                    "record for task 3 specifically — its full description, "
                    "acceptance criteria, and what it's blocked by.",
                    extra_state={"tasks": _SEED_TASKS},
                )
            )
            names = _tool_names(msg)
            if "TaskGet" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_overview_request(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "How many tasks are there in total, and what are they?",
                    extra_state={"tasks": _SEED_TASKS},
                )
            )
            names = _tool_names(msg)
            if "TaskGet" in names:
                return False, f"TaskGet fired for an overview request; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskListSelection:
    def test_selects_task_list_for_status_overview(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "What work is available right now, and where does everything stand?",
                    extra_state={"tasks": _SEED_TASKS},
                )
            )
            names = _tool_names(msg)
            if "TaskList" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_trivial_question(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(_invoke_once(kit, "What is 2 + 2?"))
            names = _tool_names(msg)
            if "TaskList" in names:
                return False, f"TaskList fired for a trivial question; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTaskStopSelection:
    def test_selects_task_stop_to_release_in_progress_task(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "I can't finish task 2 right now — put it back so someone else "
                    "can pick it up later.",
                    extra_state={"tasks": _SEED_TASKS},
                )
            )
            names = _tool_names(msg)
            if "TaskStop" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_status_request(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _tasks_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "What's the status of all the tasks right now?",
                    extra_state={"tasks": _SEED_TASKS},
                )
            )
            names = _tool_names(msg)
            if "TaskStop" in names:
                return False, f"TaskStop fired for a status request; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTeamCreateSelection:
    def test_selects_team_create_for_coordinated_multi_specialist_work(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "I've got a big cross-functional effort: research, analysis, and a "
                    "written recommendation, with specialists coordinating and handing "
                    "off to each other. Set up the people to tackle it.",
                )
            )
            names = _tool_names(msg)
            if "TeamCreate" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_trivial_question(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(_invoke_once(kit, "What is 2 + 2?"))
            names = _tool_names(msg)
            if "TeamCreate" in names:
                return False, f"TeamCreate fired for a trivial question; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTeamMessageSelection:
    def test_selects_team_message_to_assign_work(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already up and running. Have the "
                    "researcher start digging into competitor pricing.",
                )
            )
            names = _tool_names(msg)
            if "TeamMessage" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_status_check(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already running. Has the researcher "
                    "sent anything back yet?",
                )
            )
            names = _tool_names(msg)
            if "TeamMessage" in names:
                return False, f"TeamMessage fired for a status check; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTeamStatusSelection:
    def test_selects_team_status_to_collect_progress(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already running. Check whether they've "
                    "made progress and collect anything they've sent back.",
                )
            )
            names = _tool_names(msg)
            if "TeamStatus" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_when_assigning_work(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already running. Tell the researcher to "
                    "also look at the European market.",
                )
            )
            names = _tool_names(msg)
            if "TeamStatus" in names:
                return False, f"TeamStatus fired while assigning work; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestTeamDissolveSelection:
    def test_selects_team_dissolve_to_wrap_up(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already running. We're completely done — "
                    "shut the team down now.",
                )
            )
            names = _tool_names(msg)
            if "TeamDissolve" in names:
                return True, ""
            return False, f"tool_calls={names}"

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_not_selected_for_progress_update(self) -> None:
        def trial() -> tuple[bool, str]:
            kit = _team_kit()
            msg = _run(
                _invoke_once(
                    kit,
                    "The research team is already running. Ask the researcher for "
                    "a quick progress update.",
                )
            )
            names = _tool_names(msg)
            if "TeamDissolve" in names:
                return False, f"TeamDissolve fired for a progress update; tool_calls={names}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# Full-stack selection — the "harness in full glory"
#   See docs/refactor/tool-description-evals.md §8.
#
# The isolated evals above compose 1–2 extensions for clean attribution. This
# layer composes the README "recommended ordering" stack — every tool-providing
# extension, ~20 tools — so selection is tested under the realistic worst case
# where cross-extension instruction conflicts surface. A scenario that passes in
# its isolated kit but fails here is cross-tool interference. This is the
# flakiest layer (most tools → most ambiguity): read it statistically across a
# few runs, not as a single binary result.
# ---------------------------------------------------------------------------


def _recommended_kit(root: Path) -> Any:
    """Compose the README 'recommended ordering' stack, test-configured.

    Mirrors README.md's recommended composition so the eval also exercises real
    prompt-section assembly. Single-turn ``_invoke_once`` never executes tools,
    so the seeded files only need to exist for prompt realism.
    """
    from langchain_agentkit import (
        AgentsExtension,
        FilesystemExtension,
        HITLExtension,
        ResilienceExtension,
        SkillsExtension,
        TasksExtension,
        TeamExtension,
        WebSearchExtension,
    )
    from langchain_agentkit.extensions.agents import AgentConfig
    from langchain_agentkit.extensions.agents.types import _AgentConfigProxy
    from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension
    from langchain_agentkit.extensions.env.extension import EnvExtension
    from langchain_agentkit.extensions.memory import MemoryExtension
    from langchain_agentkit.extensions.skills.types import SkillConfig

    (root / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (root / "config.yaml").write_text("debug: true\nlog_level: info\n")
    (root / "MEMORY.md").write_text("# Notes\n\nThe user prefers concise answers.\n")
    src = root / "src"
    src.mkdir(exist_ok=True)
    (src / "app.py").write_text("# TODO: implement\nprint('hi')\n")

    cfg = AgentConfig(
        name="researcher",
        description="Researches topics and summarizes findings.",
        prompt="You research a topic and summarize findings concisely.",
    )
    skill = SkillConfig(
        name="market-sizing",
        description="Estimate the TAM/SAM/SOM for a market.",
        prompt="Walk through a market-sizing estimate.",
    )
    return _make_kit(
        extensions=[
            CoreBehaviorExtension(),
            MemoryExtension(path=root, project_discovery=False),
            EnvExtension(),
            FilesystemExtension(root=str(root)),
            SkillsExtension(skills=[skill]),
            TasksExtension(),
            WebSearchExtension(),
            TeamExtension(agents=[_AgentConfigProxy(cfg)]),
            AgentsExtension(agents=[cfg], default_conciseness=True),
            HITLExtension(),
            ResilienceExtension(),
        ],
        model=MODEL_NAME,
        model_resolver=lambda name: ChatOpenAI(model=name, temperature=0),
    )


# (scenario id, prompt, acceptable tools, extra_state) — same intent-prompts as
# the isolated evals, judged against the full ~20-tool stack.
_FULLSTACK_SCENARIOS = [
    ("read", "What's in the pyproject.toml file in this project?", {"Read"}, None),
    ("grep", "Find every occurrence of TODO across the files here.", {"Grep"}, None),
    ("glob", "List every Python test file in this repo.", {"Glob"}, None),
    (
        "write",
        "Create a new file called notes.txt containing the text 'Hello World'.",
        {"Write"},
        None,
    ),
    ("edit", "In config.yaml, change `debug: true` to `debug: false`.", {"Edit", "Read"}, None),
    (
        "task_create",
        "I need three things done: 1) research competitor pricing, 2) draft a "
        "proposal, 3) review it.",
        {"TaskCreate"},
        None,
    ),
    (
        "task_list",
        "What work is available right now, and where does everything stand?",
        {"TaskList"},
        {"tasks": _SEED_TASKS},
    ),
    (
        "task_get",
        "Fetch the complete stored record for task 3 — its full description and what "
        "it's blocked by.",
        {"TaskGet"},
        {"tasks": _SEED_TASKS},
    ),
    (
        "task_stop",
        "I can't finish task 2 right now — put it back so someone else can pick it up.",
        {"TaskStop"},
        {"tasks": _SEED_TASKS},
    ),
    (
        "team_create",
        "I've got a big cross-functional effort needing specialists who coordinate "
        "and hand off to each other. Set up the people for it.",
        {"TeamCreate"},
        None,
    ),
    (
        "team_message",
        "The research team is already up and running. Have the researcher start "
        "digging into competitor pricing.",
        {"TeamMessage"},
        None,
    ),
    (
        "skill",
        "Use /market-sizing to estimate the TAM for electric bikes in Europe.",
        {"Skill"},
        None,
    ),
    ("web_search", "What's the weather in Tokyo right now?", {"WebSearch"}, None),
    (
        "ask_user",
        "Process one of my files — report.csv, notes.txt, or summary.md. You must "
        "resolve which one by asking me, not guessing.",
        {"AskUser"},
        None,
    ),
    (
        "agent",
        "Delegate research on three vendors (A, B, C), one each, to run in parallel.",
        {"Agent"},
        None,
    ),
]


# Scenarios where the right tool loses to another extension's tool once the full
# stack is composed — measured cross-tool interference (gpt-5.4-mini). Tracked,
# not gated: each is a target for the description refactor (see
# docs/refactor/tool-description-evals.md §8). Flips to a hard pass once the
# losing tool's description reclaims its space.
_FULLSTACK_KNOWN_INTERFERENCE = {
    "read",  # → Glob ("what's in X" globs instead of reading)
    "grep",  # → Glob (content search globs instead of grepping)
    "team_create",  # → TaskList (Tasks tools out-compete TeamCreate)
    "team_message",  # → TeamStatus
    "ask_user",  # answers in text when many tools are present
}


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
@pytest.mark.parametrize(
    "scenario_id,prompt,expected,extra_state",
    _FULLSTACK_SCENARIOS,
    ids=[s[0] for s in _FULLSTACK_SCENARIOS],
)
def test_full_stack_selection(
    scenario_id: str,
    prompt: str,
    expected: set[str],
    extra_state: dict[str, Any] | None,
    tmp_path: Path,
) -> None:
    """Same intent-prompt, judged against the full recommended stack (~20 tools).

    Diff against the matching isolated eval: a tool that wins in isolation but
    loses here is being out-competed by another extension's tool — cross-tool
    interference. Failure messages record what actually fired.
    """

    def trial() -> tuple[bool, str]:
        kit = _recommended_kit(tmp_path)
        msg = _run(_invoke_once(kit, prompt, extra_state=extra_state))
        names = set(_tool_names(msg))
        if names & expected:
            return True, ""
        return False, f"expected one of {sorted(expected)}, got {sorted(names) or 'no tool call'}"

    passes, comments = _best_of_three(trial)
    joined = " ".join(comments)
    # The full-stack kit is token-heavy (~2K tokens/call × 20 tool schemas); run
    # spaced or it trips the per-minute token limit. Don't let that masquerade as
    # a selection miss.
    if passes < 2 and "RateLimitError" in joined:
        pytest.skip(f"[{scenario_id}] rate-limited (TPM) — run the full-stack layer spaced out")
    # Known cross-tool interference: a tracked finding, not a regression.
    if passes < 2 and scenario_id in _FULLSTACK_KNOWN_INTERFERENCE:
        pytest.xfail(f"[{scenario_id}] cross-tool interference in full stack: {comments}")
    assert passes >= 2, f"[{scenario_id}] {passes}/3 passed; {comments}"
