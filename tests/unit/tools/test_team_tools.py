"""Tests for team coordination tools."""

import asyncio
import contextlib
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
from langchain_agentkit.extensions.teams.tools import (
    _agent_team,
    _check_teammates,
    _dissolve_team,
    _require_active_team,
    _require_member,
    _send_message,
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
    return TeamExtension(agents=agents)


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
# TeamCreate
# ---------------------------------------------------------------------------


class TestTeamCreate:
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
        assert result.update["team"]["name"] == "dev-team"
        assert {m["member_name"] for m in result.update["team"]["members"]} == {
            "alice",
            "bob",
        }
        assert mw._active_team is not None
        assert mw._active_team.name == "dev-team"

        # Clean up asyncio tasks
        for task in mw._active_team.members.values():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

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
# TeamMessage
# ---------------------------------------------------------------------------


class TestTeamMessage:
    @pytest.mark.asyncio
    async def test_send_message_to_member(self):
        mw = _make_extension_with_active_team(["researcher"])

        result = await _send_message(
            to="researcher",
            message="Find information about X",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["sent_to"] == "researcher"
        assert content["message"] == "Find information about X"

        # Verify message was sent to bus
        msg = await mw._active_team.bus.receive("researcher", timeout=1.0)
        assert msg is not None
        assert msg.content == "Find information about X"

    @pytest.mark.asyncio
    async def test_send_message_broadcast(self):
        mw = _make_extension_with_active_team(["researcher", "coder"])

        result = await _send_message(
            to="*",
            message="team update",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        assert isinstance(result, Command)
        content = json.loads(result.update["messages"][0].content)
        assert content["broadcast"] is True
        assert set(content["recipients"]) == {"researcher", "coder"}

        # Verify messages were sent to all members
        msg_r = await mw._active_team.bus.receive("researcher", timeout=1.0)
        msg_c = await mw._active_team.bus.receive("coder", timeout=1.0)
        assert msg_r is not None
        assert msg_r.content == "team update"
        assert msg_c is not None
        assert msg_c.content == "team update"

    @pytest.mark.asyncio
    async def test_send_message_with_no_team_raises(self):
        mw = _make_extension_with_agents("researcher")

        with pytest.raises(ToolException, match="No active team"):
            await _send_message(
                to="researcher",
                message="hello",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_send_message_to_unknown_member_raises(self):
        mw = _make_extension_with_active_team(["researcher"])

        with pytest.raises(ToolException, match="not in active team"):
            await _send_message(
                to="nonexistent",
                message="hello",
                state={},
                tool_call_id=FAKE_TOOL_CALL_ID,
                ext=mw,
            )

    @pytest.mark.asyncio
    async def test_send_message_truncates_long_messages_in_response(self):
        mw = _make_extension_with_active_team(["researcher"])
        long_message = "x" * 200

        result = await _send_message(
            to="researcher",
            message=long_message,
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        content = json.loads(result.update["messages"][0].content)
        assert len(content["message"]) == 100

        # But full message was sent to the bus
        msg = await mw._active_team.bus.receive("researcher", timeout=1.0)
        assert msg is not None
        assert len(msg.content) == 200


# ---------------------------------------------------------------------------
# TeamStatus
# ---------------------------------------------------------------------------


class TestTeamStatus:
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
    async def test_check_teammates_includes_task_based_status(self):
        """Members with unresolved owned tasks should show as busy."""
        mw = _make_extension_with_active_team(["researcher", "coder"])

        state = {
            "tasks": [
                {
                    "id": "t1",
                    "subject": "Research X",
                    "status": "in_progress",
                    "owner": "researcher",
                },
                {
                    "id": "t2",
                    "subject": "Done task",
                    "status": "completed",
                    "owner": "coder",
                },
            ]
        }

        result = await _check_teammates(
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            ext=mw,
        )

        content = json.loads(result.update["messages"][0].content)
        members = {m["name"]: m for m in content["members"]}
        assert members["researcher"]["work_status"] == "busy"
        assert members["researcher"]["current_tasks"] == ["t1"]
        assert members["coder"]["work_status"] == "idle"
        assert members["coder"]["current_tasks"] == []

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
# TeamDissolve
# ---------------------------------------------------------------------------


class TestTeamDissolve:
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

    @pytest.mark.asyncio
    async def test_dissolve_team_unassigns_owned_tasks(self):
        """Tasks owned by dissolved teammates should reset to pending."""
        mw = _make_extension_with_active_team(["researcher"])

        # Replace with real completed task
        async def _noop():
            return "shutdown"

        for name in list(mw._active_team.members.keys()):
            task = asyncio.create_task(_noop())
            await task
            mw._active_team.members[name] = task

        state = {
            "tasks": [
                {
                    "id": "t1",
                    "subject": "Research X",
                    "status": "in_progress",
                    "owner": "researcher",
                    "active_form": "",
                },
                {
                    "id": "t2",
                    "subject": "Other work",
                    "status": "completed",
                    "owner": "researcher",
                    "active_form": "",
                },
            ]
        }

        result = await _dissolve_team(
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            timeout=5.0,
            ext=mw,
        )

        tasks = result.update.get("tasks", [])
        # t1 was in_progress — should be reset to pending with no owner
        t1 = next(t for t in tasks if t["id"] == "t1")
        assert t1["status"] == "pending"
        assert t1.get("owner") is None
        # t2 was completed — should stay completed
        t2 = next(t for t in tasks if t["id"] == "t2")
        assert t2["status"] == "completed"


# ---------------------------------------------------------------------------
# _unassign_teammate_tasks
# ---------------------------------------------------------------------------


class TestUnassignTeammateTasks:
    def test_empty_task_list(self):
        from langchain_agentkit.extensions.teams.tools import _unassign_teammate_tasks

        result = _unassign_teammate_tasks([], ["researcher"])
        assert result == []

    def test_no_owned_tasks(self):
        from langchain_agentkit.extensions.teams.tools import _unassign_teammate_tasks

        tasks = [
            {"id": "t1", "subject": "Task", "status": "pending", "owner": "other"},
            {"id": "t2", "subject": "Task", "status": "in_progress"},
        ]
        result = _unassign_teammate_tasks(tasks, ["researcher"])
        assert result[0]["owner"] == "other"
        assert result[1].get("owner") is None  # No owner, no change

    def test_preserves_completed_and_deleted(self):
        from langchain_agentkit.extensions.teams.tools import _unassign_teammate_tasks

        tasks = [
            {"id": "t1", "subject": "Done", "status": "completed", "owner": "researcher"},
            {"id": "t2", "subject": "Removed", "status": "deleted", "owner": "researcher"},
            {"id": "t3", "subject": "Active", "status": "in_progress", "owner": "researcher"},
        ]
        result = _unassign_teammate_tasks(tasks, ["researcher"])
        assert result[0]["owner"] == "researcher"  # Completed — untouched
        assert result[1]["owner"] == "researcher"  # Deleted — untouched
        assert result[2].get("owner") is None  # In progress — unassigned
        assert result[2]["status"] == "pending"

    def test_does_not_mutate_original(self):
        from langchain_agentkit.extensions.teams.tools import _unassign_teammate_tasks

        tasks = [
            {"id": "t1", "subject": "Active", "status": "in_progress", "owner": "researcher"},
        ]
        _unassign_teammate_tasks(tasks, ["researcher"])
        assert tasks[0]["owner"] == "researcher"  # Original unchanged


# ---------------------------------------------------------------------------
# _compile_with_proxy_tasks
# ---------------------------------------------------------------------------


class TestCompileWithProxyTasks:
    """Tests for predefined agent proxy tool injection."""

    def test_replaces_task_tools_with_proxies(self):
        """Predefined agent built with TasksExtension gets proxy tools instead."""
        # Build a real agent graph (uncompiled) that has TasksExtension
        from langchain_agentkit._graph_builder import build_graph
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension
        from langchain_agentkit.extensions.teams.bus import TeamMessageBus
        from langchain_agentkit.extensions.teams.tools import _compile_with_proxy_tasks

        tasks_ext = TasksExtension()
        kit = AgentKit(extensions=[tasks_ext], prompt="You are a tester.")

        async def _handler(state, *, llm, prompt, **kw):
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content="done")], "sender": "test"}

        # Need a mock LLM
        llm_mock = MagicMock()
        llm_mock.bind_tools = MagicMock(return_value=llm_mock)
        llm_mock.ainvoke = AsyncMock()

        graph = build_graph(
            name="test-agent",
            handler=_handler,
            llm=llm_mock,
            user_tools=[],
            kit=kit,
        )
        # Attach agentkit metadata (as agent metaclass does)
        graph._agentkit_handler = _handler
        graph._agentkit_llm = llm_mock
        graph._agentkit_user_tools = []
        graph._agentkit_kit = kit

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        compiled = _compile_with_proxy_tasks(graph, bus, "alice")

        # Compiled graph should exist
        assert compiled is not None
        # It should be invocable (has ainvoke)
        assert hasattr(compiled, "ainvoke")

    def test_falls_back_when_no_agentkit_metadata(self):
        """Graphs without _agentkit_* metadata compile as-is."""
        from langchain_agentkit.extensions.teams.bus import TeamMessageBus
        from langchain_agentkit.extensions.teams.tools import _compile_with_proxy_tasks

        # A plain mock StateGraph without agentkit metadata
        mock_graph = MagicMock()
        mock_graph._agentkit_handler = None
        compiled_mock = MagicMock()
        mock_graph.compile.return_value = compiled_mock

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("bob")

        result = _compile_with_proxy_tasks(mock_graph, bus, "bob")

        assert result is compiled_mock
        mock_graph.compile.assert_called_once()

    def test_preserves_non_task_user_tools(self):
        """User tools that aren't task tools are preserved in the rebuild."""
        from langchain_core.tools import StructuredTool

        from langchain_agentkit._graph_builder import build_graph
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension
        from langchain_agentkit.extensions.teams.bus import TeamMessageBus
        from langchain_agentkit.extensions.teams.tools import _compile_with_proxy_tasks

        tasks_ext = TasksExtension()
        kit = AgentKit(extensions=[tasks_ext], prompt="test")

        def _my_custom_tool(query: str) -> str:
            return "result"

        custom_tool = StructuredTool.from_function(
            func=_my_custom_tool,
            name="CustomSearch",
            description="A custom search tool",
        )

        async def _handler(state, *, llm, prompt, **kw):
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content="ok")], "sender": "t"}

        llm_mock = MagicMock()
        llm_mock.bind_tools = MagicMock(return_value=llm_mock)

        graph = build_graph(
            name="agent-with-tools",
            handler=_handler,
            llm=llm_mock,
            user_tools=[custom_tool],
            kit=kit,
        )
        graph._agentkit_handler = _handler
        graph._agentkit_llm = llm_mock
        graph._agentkit_user_tools = [custom_tool]
        graph._agentkit_kit = kit

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        compiled = _compile_with_proxy_tasks(graph, bus, "alice")

        # Should compile successfully
        assert compiled is not None
        assert hasattr(compiled, "ainvoke")
