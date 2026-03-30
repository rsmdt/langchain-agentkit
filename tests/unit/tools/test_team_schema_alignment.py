"""Tests for aligned Agent/AgentTeam tool schemas.

AgentTeam should use the same agent reference types as Agent:
- {id: "researcher"} for predefined agents
- {prompt: "You are..."} for ephemeral agents

Parameters renamed: team_name→name, members→agents, agent_type→agent.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from langgraph.types import Command

from langchain_agentkit.tools.team import _agent_team

# --- Helpers ---

FAKE_TOOL_CALL_ID = "call_test_123"


def _make_mock_agent(name: str) -> MagicMock:
    mock = MagicMock()
    mock.name = name
    mock.description = f"{name} agent"
    mock.tools_inherit = False
    mock.compile.return_value = AsyncMock()
    return mock


def _make_ext_with_agents(*names: str):
    from langchain_agentkit.extensions.teams import TeamExtension

    agents = [_make_mock_agent(n) for n in names]
    return TeamExtension(agents=agents)


# --- New schema tests ---


class TestAgentTeamNewSchema:
    """AgentTeam uses name=, agents=, agent={id:...}|{prompt:...}."""

    @pytest.mark.asyncio
    async def test_predefined_agent_via_id(self):
        ext = _make_ext_with_agents("researcher", "coder")

        result = await _agent_team(
            name="dev-team",
            agents=[
                {"name": "alice", "agent": {"id": "researcher"}},
                {"name": "bob", "agent": {"id": "coder"}},
            ],
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=ext,
        )

        assert isinstance(result, Command)
        assert result.update["team_name"] == "dev-team"
        assert ext._active_team is not None

        for task in ext._active_team.members.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    @pytest.mark.asyncio
    async def test_unknown_agent_id_raises(self):
        ext = _make_ext_with_agents("researcher")

        with pytest.raises(ToolException, match="not found"):
            await _agent_team(
                name="bad-team",
                agents=[{"name": "alice", "agent": {"id": "nonexistent"}}],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=ext,
            )

    @pytest.mark.asyncio
    async def test_duplicate_names_raises(self):
        ext = _make_ext_with_agents("researcher")

        with pytest.raises(ToolException, match="Duplicate"):
            await _agent_team(
                name="bad-team",
                agents=[
                    {"name": "alice", "agent": {"id": "researcher"}},
                    {"name": "alice", "agent": {"id": "researcher"}},
                ],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=ext,
            )

    @pytest.mark.asyncio
    async def test_ephemeral_agent_via_prompt(self):
        ext = _make_ext_with_agents("researcher")
        ext._ephemeral = True
        # Need a parent LLM for ephemeral
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="ok"))
        ext._parent_llm_getter = lambda: mock_llm

        result = await _agent_team(
            name="custom-team",
            agents=[
                {"name": "alice", "agent": {"id": "researcher"}},
                {"name": "carol", "agent": {"prompt": "You are a legal expert"}},
            ],
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=ext,
        )

        assert isinstance(result, Command)
        assert ext._active_team is not None
        # Carol should be an ephemeral team member
        assert "carol" in ext._active_team.members

        for task in ext._active_team.members.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    @pytest.mark.asyncio
    async def test_ephemeral_without_flag_raises(self):
        ext = _make_ext_with_agents("researcher")
        # ephemeral NOT enabled

        with pytest.raises(ToolException, match="[Ee]phemeral|[Dd]ynamic"):
            await _agent_team(
                name="bad-team",
                agents=[
                    {"name": "carol", "agent": {"prompt": "You are..."}},
                ],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=ext,
            )

    @pytest.mark.asyncio
    async def test_empty_agents_raises(self):
        ext = _make_ext_with_agents("researcher")

        with pytest.raises(ToolException, match="[Ee]mpty|[Cc]annot"):
            await _agent_team(
                name="empty-team",
                agents=[],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=ext,
            )


import contextlib
