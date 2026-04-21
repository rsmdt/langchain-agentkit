"""Snapshot tests for per-tool description alignment.

Each AgentKit tool's ``description`` is now carried as a module-level
constant / docstring on the tool function. Fixture files and the
``_descriptions`` loader have been removed — LangChain's convention is
that the tool function itself owns its description.

These tests verify:
- Every tool exposes a non-empty description.
- ``Bash`` description stays simplified (no git-commit/PR protocol text).
- ``Agent`` description uses domain-neutral examples.
- ``Skill`` tool description is static and does NOT carry the per-skill
  roster; the roster is appended to the system prompt by
  :meth:`SkillsExtension.prompt`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Tool discovery helpers
# ---------------------------------------------------------------------------


def _get_filesystem_tool(name: str):
    from langchain_agentkit.backends.os import OSBackend
    from langchain_agentkit.extensions.filesystem.extension import (
        FilesystemExtension,
    )

    ext = FilesystemExtension(backend=OSBackend("."))
    return next(t for t in ext.tools if t.name == name)


def _get_all_tools() -> dict[str, str]:
    """Collect (name -> description) for every tool under test."""
    from langchain_agentkit.extensions.agents.tools import create_agent_tools
    from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool
    from langchain_agentkit.extensions.skills.tools import build_skill_tool
    from langchain_agentkit.extensions.tasks.tools import create_task_tools
    from langchain_agentkit.extensions.teams.extension import TeamExtension
    from langchain_agentkit.extensions.teams.tools import create_team_tools
    from langchain_agentkit.extensions.web_search.extension import (
        WebSearchExtension,
    )

    tools: dict[str, str] = {}
    for name in ("Read", "Write", "Edit", "Glob", "Grep", "Bash"):
        tools[name] = _get_filesystem_tool(name).description

    tools["Skill"] = build_skill_tool([]).description

    ws = WebSearchExtension(providers=[])
    tools["WebSearch"] = next(t for t in ws.tools if t.name == "WebSearch").description

    tools["AskUser"] = create_ask_user_tool().description

    mock_llm = MagicMock()
    agent_tools = create_agent_tools(
        agents_by_name={},
        compiled_cache={},
        delegation_timeout=10.0,
        parent_tools_getter=list,
        ephemeral=True,
        parent_llm_getter=lambda: mock_llm,
    )
    tools["Agent"] = next(t for t in agent_tools if t.name == "Agent").description

    for t in create_task_tools():
        tools[t.name] = t.description

    stub = MagicMock()
    stub.name = "stub_agent"
    team_ext = TeamExtension(agents=[stub])
    for t in create_team_tools(team_ext):
        tools[t.name] = t.description

    return tools


_TOOL_NAMES = (
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Bash",
    "Skill",
    "WebSearch",
    "AskUser",
    "Agent",
    "TaskCreate",
    "TaskUpdate",
    "TaskList",
    "TaskGet",
    "TaskStop",
    "TeamCreate",
    "TeamMessage",
    "TeamStatus",
    "TeamDissolve",
)


# ---------------------------------------------------------------------------
# Core P0 spec tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _TOOL_NAMES)
def test_every_tool_has_description(name: str) -> None:
    tools = _get_all_tools()
    assert name in tools, f"Tool '{name}' was not built"
    desc = tools[name]
    assert desc and desc.strip(), f"Tool '{name}' has an empty description"


def test_bash_description_stays_simplified() -> None:
    """Bash description must NOT carry git-commit/PR protocol text."""
    bash_desc = _get_filesystem_tool("Bash").description
    forbidden = [
        "Committing changes with git",
        "Creating pull requests",
        "gh pr create",
        "git commit -m",
        "Git Safety Protocol",
    ]
    for phrase in forbidden:
        assert phrase not in bash_desc, (
            f"Bash description unexpectedly contains '{phrase}'; the simplified "
            f"variant must be preserved."
        )


def test_agent_description_uses_domain_neutral_examples() -> None:
    """Agent description must use domain-neutral (non-code-specific) examples."""
    tools = _get_all_tools()
    agent_desc = tools["Agent"]
    # Domain-neutral vendor/contract examples should be present.
    assert "vendor" in agent_desc.lower() or "contract" in agent_desc.lower()
    # Code-specific example markers from earlier drafts should be absent.
    forbidden = ["TypeScript", "pytest", "npm run", "ruff"]
    for phrase in forbidden:
        assert phrase not in agent_desc, f"Agent description leaks code-specific example '{phrase}'"


# ---------------------------------------------------------------------------
# Skill roster is delivered in the system prompt, not in the tool description
# ---------------------------------------------------------------------------


def test_skill_description_does_not_include_roster() -> None:
    """The Skill tool description must be static — no per-skill roster entries."""
    from langchain_agentkit.extensions.skills.tools import build_skill_tool
    from langchain_agentkit.extensions.skills.types import SkillConfig

    configs = [
        SkillConfig(name="alpha", description="Alpha skill.", prompt="alpha body"),
        SkillConfig(name="beta", description="Beta skill.", prompt="beta body"),
    ]
    tool = build_skill_tool(configs)
    assert "- alpha:" not in tool.description
    assert "- beta:" not in tool.description
    assert "Available skills:" not in tool.description
    assert "system prompt" in tool.description


def test_skill_roster_emitted_in_system_prompt() -> None:
    """SkillsExtension appends its roster to the composed system prompt."""
    from langchain_agentkit.agent_kit import AgentKit
    from langchain_agentkit.extensions.skills.extension import SkillsExtension
    from langchain_agentkit.extensions.skills.types import SkillConfig

    configs = [
        SkillConfig(name="alpha", description="Alpha skill.", prompt="alpha body"),
        SkillConfig(name="beta", description="Beta skill.", prompt="beta body"),
    ]
    kit = AgentKit(extensions=[SkillsExtension(skills=configs)])
    prompt = kit.compose({}, None).prompt
    assert "- alpha: Alpha skill." in prompt
    assert "- beta: Beta skill." in prompt


def test_skill_prompt_has_no_roster_when_no_skills() -> None:
    """An empty skill roster must not add a dangling 'Available skills:' header."""
    from langchain_agentkit.agent_kit import AgentKit
    from langchain_agentkit.extensions.skills.extension import SkillsExtension

    kit = AgentKit(extensions=[SkillsExtension(skills=[])])
    prompt = kit.compose({}, None).prompt
    assert "Available skills:" not in prompt
