"""Tests for team coordination tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException
from langgraph.types import Command

from langchain_agentkit.extensions.teams import (
    ActiveTeam,
    TeamExtension,
    TeamMessageBus,
)
from langchain_agentkit.tools.team import (
    _assign_task,
    _check_teammates,
    _dissolve_team,
    _message_teammate,
    _require_active_team,
    _require_member,
    _agent_team,
)

FAKE_TOOL_CALL_ID = "call_team_test"


def _make_mock_agent(name: str, description: str = "") -> MagicMock:
    """Create a mock agent graph with agentkit metadata."""
    mock = MagicMock()
    mock.name = name
    mock.description = description
    mock.tools_inherit = False
    mock.compile.return_value = AsyncMock()
    return mock


def _make_extension_with_agents(*names: str) -> TeamExtension:
    """Create an TeamExtension with mock agents."""
    agents = [_make_mock_agent(n) for n in names]
    return TeamExtension(agents)


def _make_extension_with_active_team(
    agent_names: list[str],
    team_name: str = "test-team",
) -> TeamExtension:
    """Create extension with mock active team."""
    mw = _make_extension_with_agents(*agent_names)

    bus = TeamMessageBus()
    bus.register("lead")

    member_tasks: dict[str, asyncio.Task] = {}
    member_types: dict[str, str] = {}

    for name in agent_names:
        bus.register(name)
        # Create a mock asyncio.Task (not a real one)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        member_tasks[name] = mock_task
        member_types[name] = name

    mw._active_team = ActiveTeam(
        name=team_name,
        bus=bus,
        members=member_tasks,
        member_types=member_types,
    )

    return mw


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestRequireActiveTeam:
    def test_raises_when_no_active_team(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            _require_active_team(mw)

    def test_returns_team_when_active(self):
        mw = _make_extension_with_active_team(["researcher"])

        team = _require_active_team(mw)

        assert team.name == "test-team"


class TestRequireMember:
    def test_raises_when_member_not_in_team(self):
        mw = _make_extension_with_active_team(["researcher"])

        with pytest.raises(ToolException, match="not in active team"):
            _require_member(mw, "nonexistent")

    def test_passes_when_member_exists(self):
        mw = _make_extension_with_active_team(["researcher"])

        _require_member(mw, "researcher")  # Should not raise


# ---------------------------------------------------------------------------
# AgentTeam
# ---------------------------------------------------------------------------


class TestAgentTeam:
    @pytest.mark.asyncio
    async def test_agent_team_creates_team(self):
        mw = _make_extension_with_agents("researcher", "coder")

        result = await _agent_team(
            name="dev-team",
            agents=[
                {"name": "alice", "agent": {"id": "researcher"}},
                {"name": "bob", "agent": {"id": "coder"}},
            ],
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        assert result.update["team_name"] == "dev-team"
        assert mw._active_team is not None
        assert mw._active_team.name == "dev-team"

        # Clean up asyncio tasks
        for task in mw._active_team.members.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_agent_team_with_duplicate_names_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="Duplicate agent names"):
            await _agent_team(
                name="bad-team",
                agents=[
                    {"name": "alice", "agent": {"id": "researcher"}},
                    {"name": "alice", "agent": {"id": "researcher"}},
                ],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_agent_team_when_team_already_active_raises(self):
        mw = _make_extension_with_active_team(["researcher"])

        with pytest.raises(ToolException, match="Team already active"):
            await _agent_team(
                name="second-team",
                agents=[{"name": "bob", "agent": {"id": "researcher"}}],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_agent_team_with_unknown_agent_type_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="not found"):
            await _agent_team(
                name="bad-team",
                agents=[{"name": "alice", "agent": {"id": "nonexistent"}}],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_agent_team_exceeding_max_size_raises(self):
        mw = _make_extension_with_agents("researcher")
        mw._max_team_size = 1

        with pytest.raises(ToolException, match="exceeds maximum"):
            await _agent_team(
                name="big-team",
                agents=[
                    {"name": "alice", "agent": {"id": "researcher"}},
                    {"name": "bob", "agent": {"id": "researcher"}},
                ],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_agent_team_with_empty_members_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="empty"):
            await _agent_team(
                name="empty-team",
                agents=[],
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )


# ---------------------------------------------------------------------------
# AssignTask
# ---------------------------------------------------------------------------


class TestAssignTask:
    @pytest.mark.asyncio
    async def test_assign_task_sends_message_to_member(self):
        mw = _make_extension_with_active_team(["researcher"])

        result = await _assign_task(
            member_name="researcher",
            task_description="Find information about X",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["sent_to"] == "researcher"
        assert content["task"] == "Find information about X"

        # No tasks in update — AssignTask only sends messages now
        assert "tasks" not in result.update

        # Verify message was sent to bus
        msg = await mw._active_team.bus.receive("researcher", timeout=1.0)
        assert msg is not None
        assert msg.content == "Find information about X"

    @pytest.mark.asyncio
    async def test_assign_task_with_no_team_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            await _assign_task(
                member_name="researcher",
                task_description="task",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_assign_task_to_unknown_member_raises(self):
        mw = _make_extension_with_active_team(["researcher"])

        with pytest.raises(ToolException, match="not in active team"):
            await _assign_task(
                member_name="nonexistent",
                task_description="task",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )


# ---------------------------------------------------------------------------
# MessageTeammate
# ---------------------------------------------------------------------------


class TestMessageTeammate:
    @pytest.mark.asyncio
    async def test_message_teammate_sends_message(self):
        mw = _make_extension_with_active_team(["researcher"])

        result = await _message_teammate(
            member_name="researcher",
            message="focus on topic Y",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["sent_to"] == "researcher"

        msg = await mw._active_team.bus.receive("researcher", timeout=1.0)
        assert msg is not None
        assert msg.content == "focus on topic Y"

    @pytest.mark.asyncio
    async def test_message_teammate_to_unknown_member_raises(self):
        mw = _make_extension_with_active_team(["researcher"])

        with pytest.raises(ToolException, match="not in active team"):
            await _message_teammate(
                member_name="nonexistent",
                message="hello",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_message_teammate_with_no_team_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            await _message_teammate(
                member_name="researcher",
                message="hello",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )


# ---------------------------------------------------------------------------
# CheckTeammates
# ---------------------------------------------------------------------------


class TestCheckTeammates:
    @pytest.mark.asyncio
    async def test_check_teammates_returns_status(self):
        mw = _make_extension_with_active_team(["researcher"])

        result = await _check_teammates(
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["team_name"] == "test-team"
        assert len(content["members"]) == 1
        assert content["members"][0]["name"] == "researcher"
        assert content["members"][0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_check_teammates_drains_lead_messages(self):
        mw = _make_extension_with_active_team(["researcher"])

        # Put a message in lead's queue
        await mw._active_team.bus.send("researcher", "lead", "I found results")

        result = await _check_teammates(
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        content = json.loads(result.update["messages"][0].content)
        assert len(content["pending_messages"]) == 1
        assert content["pending_messages"][0]["from"] == "researcher"
        assert content["pending_messages"][0]["content"] == "I found results"

    @pytest.mark.asyncio
    async def test_check_teammates_with_no_team_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            await _check_teammates(
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )


# ---------------------------------------------------------------------------
# DissolveTeam
# ---------------------------------------------------------------------------


class TestDissolveTeam:
    @pytest.mark.asyncio
    async def test_dissolve_team_shuts_down_and_clears(self):
        mw = _make_extension_with_active_team(["researcher"])

        # Replace mock tasks with real completed tasks for dissolution
        async def _noop():
            return "shutdown"

        for name in list(mw._active_team.members.keys()):
            task = asyncio.create_task(_noop())
            await task  # Let it complete
            mw._active_team.members[name] = task

        result = await _dissolve_team(
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            timeout=5.0,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["dissolved"] is True
        assert content["team_name"] == "test-team"
        assert mw._active_team is None

    @pytest.mark.asyncio
    async def test_dissolve_team_with_no_team_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            await _dissolve_team(
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_dissolve_team_returns_final_member_statuses(self):
        mw = _make_extension_with_active_team(["researcher", "coder"])

        # Replace with real completed tasks
        async def _noop():
            return "shutdown"

        for name in list(mw._active_team.members.keys()):
            task = asyncio.create_task(_noop())
            await task
            mw._active_team.members[name] = task

        result = await _dissolve_team(
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            timeout=5.0,
            ext=mw,
        )

        content = json.loads(result.update["messages"][0].content)
        member_names = [m["name"] for m in content["final_statuses"]]
        assert "researcher" in member_names
        assert "coder" in member_names
