# ruff: noqa: N801, N805
"""Tests for TeamExtension and TeamMessageBus."""

from unittest.mock import MagicMock

import pytest

from langchain_agentkit.extensions.teams import (
    TeamExtension,
    TeamMessageBus,
)
from langchain_agentkit.extensions.teams.state import TeamState


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

        mw = TeamExtension(agents=[agent_a, agent_b])

        assert "researcher" in mw._agents_by_name
        assert "coder" in mw._agents_by_name

    def test_construction_with_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            TeamExtension(agents=[])

    def test_construction_with_duplicate_names_raises_value_error(self):
        agent_a = _make_mock_agent("researcher")
        agent_b = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="Duplicate agent names"):
            TeamExtension(agents=[agent_a, agent_b])

    def test_construction_with_missing_name_raises(self):
        mock = MagicMock(spec=[])

        with pytest.raises(ValueError, match="name"):
            TeamExtension(agents=[mock])

    def test_max_team_size_validation(self):
        agent_a = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="max_team_size"):
            TeamExtension(agents=[agent_a], max_team_size=0)


class TestTeamExtensionTools:
    def test_returns_four_team_tools(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension(agents=[agent_a])
        tool_names = [t.name for t in mw.tools]

        # 4 team tools only — task tools come from TasksExtension if user adds it
        assert len(mw.tools) == 4
        assert "TeamCreate" in tool_names
        assert "TaskCreate" not in tool_names

    def test_tool_names_include_team_tools(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension(agents=[agent_a])
        tool_names = [t.name for t in mw.tools]

        assert "TeamCreate" in tool_names
        assert "TeamMessage" in tool_names
        assert "TeamStatus" in tool_names
        assert "TeamDissolve" in tool_names

    def test_tools_returns_immutable_tuple(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        first = mw.tools
        second = mw.tools

        assert first == second
        assert isinstance(first, tuple)


class TestTeamExtensionPrompt:
    def test_prompt_renders_agent_roster(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = TeamExtension(agents=[agent_a, agent_b])
        result = mw.prompt({})

        assert "researcher" in result
        assert "Research specialist" in result
        assert "coder" in result
        assert "Code specialist" in result

    def test_prompt_includes_team_coordination_guidelines(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension(agents=[agent_a])
        result = mw.prompt({})

        assert "Team Coordination" in result

    def test_prompt_shows_active_team_status_when_team_active(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

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
        mw = TeamExtension(agents=[agent_a])

        assert isinstance(mw.prompt({}), str)


class TestTeamExtensionStateSchema:
    def test_state_schema_returns_team_state(self):
        agent_a = _make_mock_agent("researcher")

        mw = TeamExtension(agents=[agent_a])
        schema = mw.state_schema

        # state_schema returns TeamState only; TasksState comes via
        # dependency resolution in AgentKit (TasksExtension auto-added)
        assert schema is TeamState


class TestTeamExtensionDependencies:
    def test_dependencies_returns_tasks_extension(self):
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        deps = mw.dependencies()

        assert len(deps) == 1
        assert isinstance(deps[0], TasksExtension)

    def test_agentkit_resolves_tasks_extension_dependency(self):
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension(agents=[agent_a])

        kit = AgentKit(extensions=[team_mw])

        # TasksExtension should be auto-added
        mw_types = [type(mw) for mw in kit._extensions]
        assert TeamExtension in mw_types
        assert TasksExtension in mw_types

    def test_agentkit_deduplicates_tasks_extension(self):
        from langchain_agentkit.agent_kit import AgentKit
        from langchain_agentkit.extensions.tasks import TasksExtension

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension(agents=[agent_a])
        tasks_mw = TasksExtension()

        # User explicitly adds TasksExtension AND TeamExtension
        kit = AgentKit(extensions=[tasks_mw, team_mw])

        # TasksExtension should NOT be duplicated
        tasks_count = sum(1 for mw in kit._extensions if isinstance(mw, TasksExtension))
        assert tasks_count == 1

    def test_agentkit_composed_schema_includes_tasks_via_dependency(self):
        from langchain_agentkit.agent_kit import AgentKit

        agent_a = _make_mock_agent("researcher")
        team_mw = TeamExtension(agents=[agent_a])

        kit = AgentKit(extensions=[team_mw])
        schema = kit.state_schema
        annotations = schema.__annotations__

        # Both TeamState and TasksState keys present via composition
        assert "team" in annotations
        assert "tasks" in annotations


class TestTeamExtensionProtocol:
    def test_satisfies_extension_protocol(self):
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)
        assert isinstance(mw.tools, (list, tuple))
        assert isinstance(mw.prompt({}), str)


# ---------------------------------------------------------------------------
# Router Node integration tests
# ---------------------------------------------------------------------------


def _extract_router_node(mw: TeamExtension) -> tuple:
    """Call graph_modifier with a mock workflow and capture the registered node."""
    registered_nodes = {}
    registered_edges = {}

    class _MockWorkflow:
        def add_node(self, name: str, func):
            registered_nodes[name] = func

        def add_conditional_edges(self, source, condition, mapping):
            registered_edges[source] = {"condition": condition, "mapping": mapping}

    mock_wf = _MockWorkflow()
    mw.graph_modifier(mock_wf, "agent")
    router_node = registered_nodes.get("router")
    router_cond = registered_edges.get("router", {}).get("condition")
    return router_node, router_cond


class TestRouterNodeTaskOps:
    """Integration tests for the router node processing task operations."""

    @pytest.mark.asyncio
    async def test_router_processes_task_create_op(self):
        """Task create op updates state['tasks'] and sends ack to teammate."""
        import json

        from langchain_agentkit.extensions.teams import ActiveTeam
        from langchain_agentkit.extensions.teams.task_proxy import TASK_OP_TYPE

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        running_task = MagicMock()
        running_task.done.return_value = False

        mw._active_team = ActiveTeam(
            name="test-team",
            bus=bus,
            members={"alice": running_task},
            member_types={"alice": "researcher"},
        )

        # Put a task create op on the lead queue
        await bus.send(
            "alice",
            "lead",
            json.dumps(
                {
                    "type": TASK_OP_TYPE,
                    "op": "create",
                    "request_id": "req-1",
                    "subject": "Write docs",
                    "description": "Document the API",
                }
            ),
        )

        router_node, _ = _extract_router_node(mw)
        result = await router_node({"tasks": []})

        # Tasks should be updated in the returned state
        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["subject"] == "Write docs"
        assert result["tasks"][0]["status"] == "pending"

        # Ack should have been sent back to alice
        ack = await bus.receive("alice", timeout=1.0)
        assert ack is not None
        parsed = json.loads(ack.content)
        assert parsed["request_id"] == "req-1"
        assert "task" in parsed

    @pytest.mark.asyncio
    async def test_router_processes_task_update_op(self):
        """Task update op modifies existing task in state."""
        import json

        from langchain_agentkit.extensions.teams import ActiveTeam
        from langchain_agentkit.extensions.teams.task_proxy import TASK_OP_TYPE

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("bob")

        running = MagicMock()
        running.done.return_value = False

        mw._active_team = ActiveTeam(
            name="t",
            bus=bus,
            members={"bob": running},
            member_types={"bob": "coder"},
        )

        await bus.send(
            "bob",
            "lead",
            json.dumps(
                {
                    "type": TASK_OP_TYPE,
                    "op": "update",
                    "request_id": "req-2",
                    "task_id": "t1",
                    "status": "completed",
                    "owner": "bob",
                }
            ),
        )

        router_node, _ = _extract_router_node(mw)
        existing_tasks = [{"id": "t1", "subject": "Fix bug", "status": "in_progress"}]
        result = await router_node({"tasks": existing_tasks})

        assert result["tasks"][0]["status"] == "completed"
        assert result["tasks"][0]["owner"] == "bob"

    @pytest.mark.asyncio
    async def test_router_mixed_task_ops_and_text(self):
        """Router separates task ops from regular messages correctly."""
        import json

        from langchain_agentkit.extensions.teams import ActiveTeam
        from langchain_agentkit.extensions.teams.task_proxy import TASK_OP_TYPE

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        running = MagicMock()
        running.done.return_value = False

        mw._active_team = ActiveTeam(
            name="t",
            bus=bus,
            members={"alice": running},
            member_types={"alice": "r"},
        )

        # Send a task op and a regular message
        await bus.send(
            "alice",
            "lead",
            json.dumps(
                {
                    "type": TASK_OP_TYPE,
                    "op": "list",
                    "request_id": "req-3",
                }
            ),
        )
        await bus.send("alice", "lead", "I finished analyzing the data")

        router_node, _ = _extract_router_node(mw)
        result = await router_node({"tasks": [{"id": "t1", "subject": "A", "status": "pending"}]})

        # Regular message becomes a HumanMessage
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "I finished analyzing" in result["messages"][0].content

        # Task list ack sent back to alice
        ack = await bus.receive("alice", timeout=1.0)
        assert ack is not None
        parsed = json.loads(ack.content)
        assert parsed["request_id"] == "req-3"

    @pytest.mark.asyncio
    async def test_router_returns_empty_when_no_team(self):
        """Router returns empty dict when no active team."""
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        router_node, _ = _extract_router_node(mw)
        result = await router_node({"tasks": []})

        assert result == {}

    @pytest.mark.asyncio
    async def test_router_returns_empty_when_no_messages_and_no_active_members(self):
        """Router returns empty when bus is empty and all members are done."""
        from langchain_agentkit.extensions.teams import ActiveTeam

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a], router_timeout=0.1)

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        done_task = MagicMock()
        done_task.done.return_value = True

        mw._active_team = ActiveTeam(
            name="t",
            bus=bus,
            members={"alice": done_task},
            member_types={"alice": "r"},
        )

        router_node, _ = _extract_router_node(mw)
        result = await router_node({"tasks": []})

        assert result == {}

    @pytest.mark.asyncio
    async def test_router_should_continue_returns_end_when_no_team(self):
        """Condition function routes to END when no active team."""
        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        _, router_cond = _extract_router_node(mw)
        result = router_cond({"messages": []})

        assert result == "__end__"  # langgraph END sentinel

    @pytest.mark.asyncio
    async def test_router_should_continue_routes_to_agent_on_teammate_message(self):
        """Condition function routes to agent node when teammate message present."""
        from langchain_core.messages import HumanMessage

        from langchain_agentkit.extensions.teams import ActiveTeam

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        bus = TeamMessageBus()
        bus.register("lead")

        running = MagicMock()
        running.done.return_value = False

        mw._active_team = ActiveTeam(
            name="t",
            bus=bus,
            members={"a": running},
            member_types={"a": "r"},
        )

        _, router_cond = _extract_router_node(mw)
        state = {
            "messages": [
                HumanMessage(content="result"),
            ],
        }
        result = router_cond(state)

        assert result == "agent"  # routes back to the agent node


# ---------------------------------------------------------------------------
# Router ↔ run-lifecycle wiring
# ---------------------------------------------------------------------------


def _extract_router_mapping(
    mw: TeamExtension,
    workflow_nodes: dict | None = None,
) -> tuple:
    """Call graph_modifier with a mock workflow carrying ``nodes``.

    Returns ``(condition_fn, destination_mapping)``.
    """
    registered_nodes: dict = {}
    registered_edges: dict = {}

    class _MockWorkflow:
        def __init__(self) -> None:
            self.nodes = dict(workflow_nodes or {})

        def add_node(self, name: str, func):
            registered_nodes[name] = func
            self.nodes[name] = func

        def add_conditional_edges(self, source, condition, mapping):
            registered_edges[source] = {"condition": condition, "mapping": mapping}

    mock_wf = _MockWorkflow()
    mw.graph_modifier(mock_wf, "agent")
    cond = registered_edges.get("router", {}).get("condition")
    mapping = registered_edges.get("router", {}).get("mapping")
    return cond, mapping


class TestRouterRunExitWiring:
    """Router routes terminating edges to ``_run_exit`` when run hooks are wired.

    Without this, the team router jumps straight to ``END`` and bypasses
    ``_run_exit``, which means ``after_run`` hooks never fire and any
    runtime cleanup registered there (teammate tasks, bus, capture buffer)
    is silently skipped.
    """

    def test_router_targets_run_exit_when_present(self):
        """If _run_exit is in the workflow, router destinations reference it."""
        from langgraph.graph import END

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        cond, mapping = _extract_router_mapping(
            mw,
            workflow_nodes={"_run_exit": lambda s: s},
        )

        assert "_run_exit" in mapping
        assert mapping["_run_exit"] == "_run_exit"
        assert END not in mapping

    def test_router_falls_back_to_end_without_run_exit(self):
        """If _run_exit is absent, router destinations use END."""
        from langgraph.graph import END

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        cond, mapping = _extract_router_mapping(mw, workflow_nodes={})

        assert END in mapping
        assert "_run_exit" not in mapping

    def test_router_should_continue_returns_run_exit_when_wired(self):
        """When _run_exit is wired, _router_should_continue returns that label."""
        from langchain_agentkit.extensions.teams import ActiveTeam

        agent_a = _make_mock_agent("researcher")
        mw = TeamExtension(agents=[agent_a])

        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("a")
        done_task = MagicMock()
        done_task.done.return_value = True
        done_task.cancelled.return_value = False
        done_task.exception.return_value = None

        mw._active_team = ActiveTeam(
            name="t",
            bus=bus,
            members={"a": done_task},
            member_types={"a": "r"},
        )

        cond, _ = _extract_router_mapping(
            mw,
            workflow_nodes={"_run_exit": lambda s: s},
        )

        # No teammate message, no active members — should terminate via
        # _run_exit, not END.
        result = cond({"messages": []})
        assert result == "_run_exit"

    def test_after_run_fires_on_router_termination_path(self):
        """End-to-end: a TeamExtension + after_run hook sees after_run invoked.

        Build a real agent with both TeamExtension and an extension that
        defines ``after_run``.  With no active team and no tool calls, the
        agent node's own ``_should_continue`` handles termination and
        ``after_run`` fires.  The critical property is that the router's
        graph-modifier step did not break the build by referencing a
        missing ``_run_exit`` node.
        """
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage, HumanMessage

        from langchain_agentkit import agent
        from langchain_agentkit.extension import Extension

        after_run_calls: list[str] = []

        class LifecycleExtension(Extension):
            async def after_run(self, *, state, runtime):
                after_run_calls.append("after_run")
                return None

        agent_a = _make_mock_agent("researcher")

        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        class my_agent(agent):
            model = mock_llm
            extensions = [
                TeamExtension(agents=[agent_a]),
                LifecycleExtension(),
            ]

            async def handler(state, *, llm, tools, prompt):
                bound = llm.bind_tools(tools)
                response = await bound.ainvoke(state["messages"])
                return {"messages": [response]}

        compiled = my_agent.compile()

        # Both lifecycle nodes and router should be present in the graph.
        assert "_run_exit" in my_agent.nodes
        assert "_run_entry" in my_agent.nodes
        assert "router" in my_agent.nodes

        import asyncio

        asyncio.run(compiled.ainvoke({"messages": [HumanMessage(content="hi")]}))

        assert after_run_calls == ["after_run"]


class TestKitSetupWiring:
    """setup() picks up kit-level llm_getter / tools_getter.

    Without this wiring, teammates that inherit the parent LLM/tools
    crash at tool-call time because ``_parent_llm_getter`` is None.
    """

    _AGENT_MD = """\
---
name: researcher
description: Research specialist
---
You are a research assistant.
"""

    async def test_setup_wires_llm_getter(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(self._AGENT_MD)
            ext = TeamExtension(agents=tmpdir)

            sentinel_llm = object()
            await ext.setup(extensions=[ext], llm_getter=lambda: sentinel_llm)

            assert ext._parent_llm_getter is not None
            assert ext._parent_llm_getter() is sentinel_llm

    async def test_setup_wires_tools_getter(self):
        import tempfile
        from pathlib import Path

        from langchain_agentkit.extensions.teams.extension import (
            _default_tools_getter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(self._AGENT_MD)
            ext = TeamExtension(agents=tmpdir)

            sentinel_tools: list = []
            await ext.setup(extensions=[ext], tools_getter=lambda: sentinel_tools)

            assert ext._parent_tools_getter is not _default_tools_getter
            assert ext._parent_tools_getter() is sentinel_tools

    async def test_setup_does_not_override_explicit_getters(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(self._AGENT_MD)
            ext = TeamExtension(agents=tmpdir)

            explicit_llm = object()
            explicit_tools: list = []
            ext.set_parent_llm_getter(lambda: explicit_llm)
            ext.set_parent_tools_getter(lambda: explicit_tools)

            await ext.setup(
                extensions=[ext],
                llm_getter=lambda: "kit-level",
                tools_getter=lambda: ["kit-level"],
            )

            assert ext._parent_llm_getter() is explicit_llm
            assert ext._parent_tools_getter() is explicit_tools

    async def test_run_extension_setup_wires_kit_model_into_team(self):
        """Regression: config-based teammates used to crash at tool-call time.

        Before kit-level wiring in ``run_extension_setup``, the
        TeamExtension had ``_parent_llm_getter=None`` and teammate
        spawning at ``extension.py:508`` raised ``TypeError: NoneType
        object is not callable``.
        """
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from langchain_agentkit.agent_kit import AgentKit, run_extension_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(self._AGENT_MD)
            ext = TeamExtension(agents=tmpdir)

            fake_llm = MagicMock(name="fake_llm")
            kit = AgentKit(extensions=[ext], model=fake_llm)

            await run_extension_setup(kit)

            assert ext._parent_llm_getter is not None, (
                "kit-level llm_getter was not wired into TeamExtension"
            )
            assert ext._parent_llm_getter() is fake_llm
            assert ext._parent_tools_getter() == list(kit.tools)
