"""Tests for TeamExtension path/backend discovery and AgentConfig teammates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from langchain_agentkit.extensions.agents.types import AgentConfig, _AgentConfigProxy
from langchain_agentkit.extensions.teams import TeamExtension

if TYPE_CHECKING:
    from pathlib import Path


def _agent_md(name: str, description: str, body: str = "Do the work.") -> str:
    return f"""---
name: {name}
description: {description}
---
{body}
"""


def test_path_mode_discovers_local_directory(tmp_path: Path) -> None:
    (tmp_path / "researcher.md").write_text(_agent_md("researcher", "Researches things."))
    (tmp_path / "writer.md").write_text(_agent_md("writer", "Writes things."))

    ext = TeamExtension(agents=tmp_path)

    assert set(ext.agents_by_name.keys()) == {"researcher", "writer"}
    for proxy in ext.agents_by_name.values():
        assert isinstance(proxy, _AgentConfigProxy)


def test_path_mode_with_backend_defers_discovery(tmp_path: Path) -> None:
    backend = MagicMock()
    ext = TeamExtension(agents=tmp_path, backend=backend)

    # Nothing discovered until setup() runs.
    assert ext.agents_by_name == {}
    assert ext._deferred_path == str(tmp_path)


@pytest.mark.asyncio
async def test_backend_setup_populates_roster(tmp_path: Path) -> None:
    """setup() runs async backend discovery when backend+path is provided."""
    from langchain_agentkit.extensions.agents import discovery as agents_discovery

    async def fake_discover(_backend: Any, _path: str) -> list[AgentConfig]:
        return [
            AgentConfig(name="analyst", description="Analyzes data.", prompt="Analyst prompt."),
        ]

    backend = MagicMock()
    ext = TeamExtension(agents=tmp_path, backend=backend)

    # Patch the specific helper the extension imports inside setup().
    original = agents_discovery.discover_agents_from_backend
    agents_discovery.discover_agents_from_backend = fake_discover  # type: ignore[assignment]
    try:
        await ext.setup(extensions=[ext])
    finally:
        agents_discovery.discover_agents_from_backend = original  # type: ignore[assignment]

    assert set(ext.agents_by_name.keys()) == {"analyst"}
    assert ext._deferred_path is None


def test_invalid_agents_type_raises() -> None:
    with pytest.raises(TypeError, match="agents must be"):
        TeamExtension(agents=123)  # type: ignore[arg-type]


def test_compile_config_with_proxy_tasks_resolves_model_tools_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config-based teammate path resolves model, tools, skills and proxy tasks."""
    from langchain_core.tools import StructuredTool

    from langchain_agentkit.extensions.teams import tools as teams_tools_pkg  # noqa: F401
    from langchain_agentkit.extensions.teams.bus import TeamMessageBus
    from langchain_agentkit.extensions.teams.tools import shared as shared_mod

    mock_llm = MagicMock()

    def _noop(x: str) -> str:
        """Test tool."""
        return x

    allowed = StructuredTool.from_function(_noop, name="Search")
    denied = StructuredTool.from_function(_noop, name="Delete")
    task_tool = StructuredTool.from_function(_noop, name="TaskCreate")

    resolved_skill = "Extra skill content."

    config = AgentConfig(
        name="analyst",
        description="Analyst.",
        prompt="You analyze.",
        tools=["Search", "TaskCreate"],  # TaskCreate must be stripped; Search kept.
        model="gpt-test",
        skills=["analysis-tips"],
    )

    bus = TeamMessageBus()
    bus.register("analyst")

    captured: dict[str, Any] = {}

    def _fake_build(
        *, name: str, llm: Any, prompt: str, user_tools: list[Any], max_turns: Any = None
    ) -> Any:
        captured["name"] = name
        captured["llm"] = llm
        captured["prompt"] = prompt
        captured["user_tools"] = user_tools
        captured["max_turns"] = max_turns
        return MagicMock(name="compiled-graph")

    monkeypatch.setattr("langchain_agentkit.graph_builder.build_ephemeral_graph", _fake_build)

    compiled = shared_mod._compile_config_with_proxy_tasks(
        config,
        bus,
        "analyst",
        parent_tools_getter=lambda: [allowed, denied, task_tool],
        parent_llm_getter=None,
        model_resolver=lambda _m: mock_llm,
        skills_resolver=lambda _n: resolved_skill,
    )

    assert compiled is not None
    assert captured["name"] == "analyst"
    assert captured["llm"] is mock_llm
    # Skill content concatenated and teammate addendum prepended.
    assert resolved_skill in captured["prompt"]
    assert "You analyze." in captured["prompt"]
    assert "Agent Teammate Communication" in captured["prompt"]
    # Tool filtering: Search kept, Delete filtered, lead's TaskCreate
    # stripped and replaced by bus-proxied TaskCreate/Update/List/Get.
    user_tools = captured["user_tools"]
    tool_names = [t.name for t in user_tools]
    assert "Search" in tool_names
    assert "Delete" not in tool_names
    # TaskCreate should appear exactly once — the proxy, not the lead's.
    assert tool_names.count("TaskCreate") == 1
    assert user_tools[-1].name.startswith("Task")  # proxy tools appended last
    for proxy_name in ("TaskCreate", "TaskUpdate", "TaskList", "TaskGet"):
        assert proxy_name in tool_names


def test_compile_config_raises_without_model_source() -> None:
    from langchain_core.tools import ToolException

    from langchain_agentkit.extensions.teams.bus import TeamMessageBus
    from langchain_agentkit.extensions.teams.tools.shared import (
        _compile_config_with_proxy_tasks,
    )

    config = AgentConfig(name="a", description="d", prompt="p")
    bus = TeamMessageBus()
    bus.register("a")

    with pytest.raises(ToolException, match="requires a model"):
        _compile_config_with_proxy_tasks(
            config,
            bus,
            "a",
            parent_tools_getter=list,
            parent_llm_getter=None,
            model_resolver=None,
            skills_resolver=None,
        )
