"""Tests for TeamExtension and TeamMessageBus."""

import asyncio
from unittest.mock import MagicMock

import pytest

from langchain_agentkit.extensions.teams import (
    TeamExtension,
    TeamMessageBus,
)
from langchain_agentkit.state import TeamState


def _make_mock_agent(name: str, description: str = "") -> MagicMock:
    """Create a mock agent graph with agentkit metadata."""
    mock = MagicMock()
    mock.name = name
    mock.description = description
    mock.tools_inherit = False
    mock.compile.return_value = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# TeamMessageBus tests
# ---------------------------------------------------------------------------


class TestTeamMessageBusRegistration:
    def test_register_creates_queue(self):
        bus = TeamMessageBus()

        bus.register("alice")

        assert "alice" in bus._queues

    def test_register_is_idempotent(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("alice")

        assert "alice" in bus._queues

    def test_unregister_removes_queue(self):
        bus = TeamMessageBus()
        bus.register("alice")

        bus.unregister("alice")

        assert "alice" not in bus._queues

    def test_unregister_nonexistent_is_safe(self):
        bus = TeamMessageBus()

        bus.unregister("nonexistent")  # Should not raise


class TestTeamMessageBusSend:
    @pytest.mark.asyncio
    async def test_send_message_received_by_target(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("bob")

        await bus.send("alice", "bob", "hello bob")

        msg = await bus.receive("bob", timeout=1.0)
        assert msg is not None
        assert msg.sender == "alice"
        assert msg.receiver == "bob"
        assert msg.content == "hello bob"

    @pytest.mark.asyncio
    async def test_send_to_unregistered_agent_raises_value_error(self):
        bus = TeamMessageBus()
        bus.register("alice")

        with pytest.raises(ValueError, match="not registered"):
            await bus.send("alice", "unknown", "hello")

    @pytest.mark.asyncio
    async def test_fifo_ordering_preserved(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("bob")

        await bus.send("alice", "bob", "first")
        await bus.send("alice", "bob", "second")
        await bus.send("alice", "bob", "third")

        msg1 = await bus.receive("bob", timeout=1.0)
        msg2 = await bus.receive("bob", timeout=1.0)
        msg3 = await bus.receive("bob", timeout=1.0)

        assert msg1.content == "first"
        assert msg2.content == "second"
        assert msg3.content == "third"


class TestTeamMessageBusBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_except_sender(self):
        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")
        bus.register("bob")

        await bus.broadcast("lead", "team update")

        msg_alice = await bus.receive("alice", timeout=1.0)
        msg_bob = await bus.receive("bob", timeout=1.0)
        msg_lead = await bus.receive("lead", timeout=0.1)

        assert msg_alice is not None
        assert msg_alice.content == "team update"
        assert msg_bob is not None
        assert msg_bob.content == "team update"
        assert msg_lead is None  # Sender does not receive


class TestTeamMessageBusReceive:
    @pytest.mark.asyncio
    async def test_receive_with_timeout_returns_none(self):
        bus = TeamMessageBus()
        bus.register("alice")

        msg = await bus.receive("alice", timeout=0.05)

        assert msg is None

    @pytest.mark.asyncio
    async def test_receive_for_unregistered_returns_none(self):
        bus = TeamMessageBus()

        msg = await bus.receive("nonexistent", timeout=0.05)

        assert msg is None


class TestTeamMessageBusPendingCount:
    @pytest.mark.asyncio
    async def test_pending_count_accurate(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("bob")

        assert bus.pending_count("bob") == 0

        await bus.send("alice", "bob", "msg1")
        assert bus.pending_count("bob") == 1

        await bus.send("alice", "bob", "msg2")
        assert bus.pending_count("bob") == 2

        await bus.receive("bob", timeout=1.0)
        assert bus.pending_count("bob") == 1

    def test_pending_count_unregistered_returns_zero(self):
        bus = TeamMessageBus()

        assert bus.pending_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# TeamExtension tests
# ---------------------------------------------------------------------------


class TestTeamExtensionConstruction:
    def test_construction_with_valid_agents(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = TeamExtension([agent_a, agent_b])

        assert "researcher" in mw._agents_by_name
        assert "coder" in mw._agents_by_name

    def test_construction_with_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            TeamExtension([])

    def test_construction_with_duplicate_names_raises_value_error(self):
        agent_a = _make_mock_agent("researcher")
        agent_b = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="Duplicate agent names"):
            TeamExtension([agent_a, agent_b])

    def test_construction_with_missing_name_raises(self):
        mock = MagicMock(spec=[])

        with pytest.raises(ValueError, match="name"):
            TeamExtension([mock])

    def test_max_team_size_validation(self):
        agent_a = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="max_team_size"):
            TeamExtension([agent_a], max_team_size=0)


class TestTeamExtensionTools:
    def test_returns_five_team_tools(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension([agent_a])
        tool_names = [t.name for t in mw.tools]

        # 5 team tools only — task tools come from TasksExtension if user adds it
        assert len(mw.tools) == 5
        assert "AgentTeam" in tool_names
        assert "TaskCreate" not in tool_names

    def test_tool_names_include_team_tools(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension([agent_a])
        tool_names = [t.name for t in mw.tools]

        assert "AgentTeam" in tool_names
        assert "AssignTask" in tool_names
        assert "MessageTeammate" in tool_names
        assert "CheckTeammates" in tool_names
        assert "DissolveTeam" in tool_names

    def test_tools_returns_immutable_tuple(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension([agent_a])

        first = mw.tools
        second = mw.tools

        assert first == second
        assert isinstance(first, tuple)


class TestTeamExtensionPrompt:
    def test_prompt_renders_agent_roster(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = TeamExtension([agent_a, agent_b])
        result = mw.prompt({})

        assert "researcher" in result
        assert "Research specialist" in result
        assert "coder" in result
        assert "Code specialist" in result

    def test_prompt_includes_team_coordination_guidelines(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension([agent_a])
        result = mw.prompt({})

        assert "Team Coordination" in result

    def test_prompt_shows_active_team_status_when_team_active(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension([agent_a])

        # Simulate an active team
        from langchain_agentkit.extensions.teams import ActiveTeam

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        # Create a mock completed task
        mock_task = MagicMock()
        mock_task.done.return_value = False

        mw._active_team = ActiveTeam(
            name="test-team",
            bus=bus,
            members={"alice": mock_task},
            member_types={"alice": "researcher"},
        )

        result = mw.prompt({})

        assert "Active Team: test-team" in result
        assert "alice" in result

    def test_prompt_returns_string(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension([agent_a])

        assert isinstance(mw.prompt({}), str)


class TestTeamExtensionStateSchema:
    def test_state_schema_returns_team_state(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension([agent_a])
        schema = mw.state_schema

        # state_schema returns TeamState only; TasksState comes via
        # dependency resolution in AgentKit (TasksExtension auto-added)
        assert schema is TeamState


class TestTeamExtensionDependencies:
    def test_dependencies_returns_tasks_extension(self):
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension([agent_a])

        deps = mw.dependencies()

        assert len(deps) == 1
        assert isinstance(deps[0], TasksExtension)

    def test_agentkit_resolves_tasks_extension_dependency(self):
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension([agent_a])

        kit = AgentKit([team_mw])

        # TasksExtension should be auto-added
        mw_types = [type(mw) for mw in kit._extensions]
        assert TeamExtension in mw_types
        assert TasksExtension in mw_types

    def test_agentkit_deduplicates_tasks_extension(self):
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension([agent_a])
        tasks_mw = TasksExtension()

        # User explicitly adds TasksExtension AND TeamExtension
        kit = AgentKit([tasks_mw, team_mw])

        # TasksExtension should NOT be duplicated
        tasks_count = sum(1 for mw in kit._extensions if isinstance(mw, TasksExtension))
        assert tasks_count == 1

    def test_agentkit_composed_schema_includes_tasks_via_dependency(self):
        from langchain_agentkit.agent_kit import AgentKit

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension([agent_a])

        kit = AgentKit([team_mw])
        schema = kit.state_schema
        annotations = schema.__annotations__

        # Both TeamState and TasksState keys present via composition
        assert "team_members" in annotations
        assert "tasks" in annotations


class TestTeamExtensionProtocol:
    def test_satisfies_extension_protocol(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension([agent_a])

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)
        assert isinstance(mw.tools, (list, tuple))
        assert isinstance(mw.prompt({}), str)
