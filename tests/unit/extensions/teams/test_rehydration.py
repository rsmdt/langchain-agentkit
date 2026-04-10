"""Tests for cross-turn rehydration of ``TeamExtension``.

Covers:
- ``rehydrate_if_needed`` is a no-op when state has no team
- Rehydrating from a ``TeammateSpec`` list rebuilds bus + asyncio.Tasks
- Tagged messages in ``state["messages"]`` seed each member's history
- ``before_model`` hook drains the capture buffer into persisted state
- ``wrap_model`` filter hides team-tagged messages from the lead
- ``after_run`` cleans up but does NOT clear ``state["team"]``
- Dissolve clears ``state["team"]`` and capture buffer
- Rehydration is idempotent under concurrent calls
- Missing roster agent produces a degraded slot
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.extensions.teams import TeamExtension
from langchain_agentkit.extensions.teams.filter import TEAM_KEY, team_member_of


def _make_mock_agent(name: str) -> MagicMock:
    mock = MagicMock()
    mock.name = name
    mock.description = f"{name} description"
    mock.tools_inherit = False
    return mock


def _patch_build_teammate_graph(ext: TeamExtension, record_name: list[tuple] | None = None) -> None:
    """Replace ``build_teammate_graph`` with one that returns a no-op graph.

    Each rebuilt teammate gets a fresh mock whose ``ainvoke`` echoes
    ``{"messages": state["messages"] + [AIMessage("ok")]}`` — satisfies
    the append-reducer precondition in ``_teammate_loop``.
    """

    def _mock_build(spec: Any, bus: Any) -> Any:
        if record_name is not None:
            record_name.append((spec["member_name"], spec["kind"]))

        async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
            return {
                "messages": list(state["messages"]) + [AIMessage(content="ok")],
                "sender": spec["member_name"],
            }

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=_ainvoke)
        return graph

    ext.build_teammate_graph = _mock_build  # type: ignore[method-assign]


async def _shutdown_extension(ext: TeamExtension) -> None:
    if ext._active_team is None:
        return
    for task in list(ext._active_team.members.values()):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    ext._active_team = None


class TestRehydrateIfNeeded:
    @pytest.mark.asyncio
    async def test_no_team_is_noop(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        await ext.rehydrate_if_needed({"messages": []})

        assert ext._active_team is None

    @pytest.mark.asyncio
    async def test_empty_state_is_noop(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        await ext.rehydrate_if_needed({})

        assert ext._active_team is None

    @pytest.mark.asyncio
    async def test_rebuilds_from_team_metadata(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        built: list[tuple] = []
        _patch_build_teammate_graph(ext, built)

        state = {
            "team": {
                "name": "research-team",
                "members": [
                    {
                        "member_name": "r1",
                        "kind": "predefined",
                        "agent_id": "researcher",
                    },
                ],
            },
            "messages": [],
        }

        try:
            await ext.rehydrate_if_needed(state)

            assert ext._active_team is not None
            assert ext._active_team.name == "research-team"
            assert "r1" in ext._active_team.members
            assert built == [("r1", "predefined")]
        finally:
            await _shutdown_extension(ext)

    @pytest.mark.asyncio
    async def test_idempotent_on_repeated_calls(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        _patch_build_teammate_graph(ext)

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [],
        }

        try:
            await ext.rehydrate_if_needed(state)
            first_team = ext._active_team

            await ext.rehydrate_if_needed(state)

            assert ext._active_team is first_team
        finally:
            await _shutdown_extension(ext)

    @pytest.mark.asyncio
    async def test_concurrent_rehydration_is_serialized(self):
        """Parallel rehydrate calls must not race — lock + re-check."""
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        build_count = {"n": 0}

        def _slow_build(spec: Any, bus: Any) -> Any:
            build_count["n"] += 1

            async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
                return {"messages": list(state["messages"]), "sender": "r1"}

            graph = MagicMock()
            graph.ainvoke = AsyncMock(side_effect=_ainvoke)
            return graph

        ext.build_teammate_graph = _slow_build  # type: ignore[method-assign]

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [],
        }

        try:
            await asyncio.gather(
                ext.rehydrate_if_needed(state),
                ext.rehydrate_if_needed(state),
                ext.rehydrate_if_needed(state),
            )

            # Build was called exactly once — subsequent callers saw
            # the populated _active_team inside the lock and exited.
            assert build_count["n"] == 1
            assert ext._active_team is not None
            assert len(ext._active_team.members) == 1
        finally:
            await _shutdown_extension(ext)

    @pytest.mark.asyncio
    async def test_missing_roster_agent_yields_degraded_slot(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        # Don't patch build_teammate_graph — real one will raise KeyError
        # because "ghost" isn't in the roster.

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "ghost", "kind": "predefined", "agent_id": "missing"},
                ],
            },
            "messages": [],
        }

        try:
            await ext.rehydrate_if_needed(state)

            assert ext._active_team is not None
            assert "ghost" in ext._active_team.members
            label = ext._active_team.member_types["ghost"]
            assert label.startswith("unavailable:")
        finally:
            await _shutdown_extension(ext)


class TestRehydrationWithTaggedHistory:
    @pytest.mark.asyncio
    async def test_history_filtered_by_team_member(self):
        """Each rebuilt teammate is seeded with ONLY its tagged slice."""
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        seen_histories: dict[str, list[Any]] = {}

        def _build(spec: Any, bus: Any) -> Any:
            name = spec["member_name"]

            async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
                # Snapshot what was passed in on first invocation.
                if name not in seen_histories:
                    seen_histories[name] = list(state["messages"])
                return {
                    "messages": list(state["messages"]) + [AIMessage(content="ok")],
                    "sender": name,
                }

            graph = MagicMock()
            graph.ainvoke = AsyncMock(side_effect=_ainvoke)
            return graph

        ext.build_teammate_graph = _build  # type: ignore[method-assign]

        # State carries r1's and r2's prior-turn messages plus lead's own.
        r1_human = HumanMessage(
            content="r1 task",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        r1_ai = AIMessage(
            content="r1 done",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
            name="r1",
        )
        r2_human = HumanMessage(
            content="r2 task",
            additional_kwargs={TEAM_KEY: {"member": "r2"}},
        )
        lead_human = HumanMessage(content="user")

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                    {"member_name": "r2", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [lead_human, r1_human, r2_human, r1_ai],
        }

        try:
            await ext.rehydrate_if_needed(state)

            # Send a follow-up to r1 and r2 to drive their loops.
            bus = ext._active_team.bus
            await bus.send("lead", "r1", "continue")
            await bus.send("lead", "r2", "continue")
            await bus.receive("lead", timeout=2.0)
            await bus.receive("lead", timeout=2.0)

            # r1's invocation should have seen its two prior messages
            # PLUS the new "continue" HumanMessage (3 total).
            assert len(seen_histories["r1"]) == 3
            assert seen_histories["r1"][0] is r1_human
            assert seen_histories["r1"][1] is r1_ai
            assert seen_histories["r1"][2].content == "continue"

            # r2's invocation saw its one prior message + new one (2 total).
            assert len(seen_histories["r2"]) == 2
            assert seen_histories["r2"][0] is r2_human
            assert seen_histories["r2"][1].content == "continue"
        finally:
            await _shutdown_extension(ext)


class TestBeforeModelFlushesCaptureBuffer:
    @pytest.mark.asyncio
    async def test_before_model_drains_buffer(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        # Pre-populate the capture buffer.
        tagged = HumanMessage(
            content="from a teammate",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        ext._capture_buffer.append(tagged)

        result = await ext.before_model(state={"messages": []}, runtime=None)

        assert result == {"messages": [tagged]}
        assert ext._capture_buffer == []

    @pytest.mark.asyncio
    async def test_before_model_empty_buffer_returns_none(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        result = await ext.before_model(state={"messages": []}, runtime=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_before_model_rehydrates_and_flushes(self):
        """On turn 2, before_model both rehydrates AND drains the (empty) buffer."""
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        _patch_build_teammate_graph(ext)

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [],
        }

        try:
            result = await ext.before_model(state=state, runtime=None)

            assert ext._active_team is not None
            assert "r1" in ext._active_team.members
            # Buffer was empty so nothing to flush.
            assert result is None
        finally:
            await _shutdown_extension(ext)


class TestWrapModelFilter:
    @pytest.mark.asyncio
    async def test_filters_team_messages_before_handler(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        lead_user = HumanMessage(content="user")
        tagged = AIMessage(
            content="hidden",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        lead_ai = AIMessage(content="visible")

        seen_in_handler: dict[str, Any] = {}

        async def _handler(state: dict[str, Any]) -> dict[str, Any]:
            seen_in_handler["messages"] = list(state["messages"])
            return {}

        await ext.wrap_model(
            state={"messages": [lead_user, tagged, lead_ai]},
            handler=_handler,
            runtime=None,
        )

        # Handler saw only the untagged messages.
        assert seen_in_handler["messages"] == [lead_user, lead_ai]

    @pytest.mark.asyncio
    async def test_state_is_not_mutated(self):
        """The filter is non-destructive — original state is untouched."""
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        tagged = AIMessage(
            content="hidden",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        original = [HumanMessage(content="user"), tagged]
        state = {"messages": list(original)}

        async def _handler(_s: dict[str, Any]) -> dict[str, Any]:
            return {}

        await ext.wrap_model(state=state, handler=_handler, runtime=None)

        # Original list unchanged.
        assert state["messages"] == original


class TestAfterRunCleanup:
    @pytest.mark.asyncio
    async def test_after_run_flushes_remaining_buffer(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        tagged = AIMessage(
            content="late",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        ext._capture_buffer.append(tagged)

        result = await ext.after_run(state={}, runtime=None)

        assert result == {"messages": [tagged]}
        assert ext._capture_buffer == []

    @pytest.mark.asyncio
    async def test_after_run_does_not_clear_team_metadata(self):
        """Turn-end cleanup must not return ``team: None`` — rehydration needs it."""
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])

        result = await ext.after_run(state={}, runtime=None)

        # Either None (no flush) or a dict with only messages — never 'team'.
        if result is not None:
            assert "team" not in result

    @pytest.mark.asyncio
    async def test_after_run_clears_runtime_state(self):
        ext = TeamExtension(agents=[_make_mock_agent("researcher")])
        _patch_build_teammate_graph(ext)

        state = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [],
        }
        await ext.rehydrate_if_needed(state)
        assert ext._active_team is not None

        await ext.after_run(state=state, runtime=None)

        assert ext._active_team is None


class TestTwoTurnScenario:
    @pytest.mark.asyncio
    async def test_turn1_captures_flushed_turn2_rehydrates(self):
        """End-to-end: capture happens turn 1, state carries tags, rehydrate turn 2."""
        # --- Turn 1 ---
        ext1 = TeamExtension(agents=[_make_mock_agent("researcher")])
        _patch_build_teammate_graph(ext1)

        # Simulate what the refactor's before_model does: rehydrate + flush.
        state_turn1 = {
            "team": {
                "name": "t",
                "members": [
                    {"member_name": "r1", "kind": "predefined", "agent_id": "researcher"},
                ],
            },
            "messages": [HumanMessage(content="user")],
        }
        try:
            await ext1.rehydrate_if_needed(state_turn1)
            bus = ext1._active_team.bus

            # Drive the teammate once
            await bus.send("lead", "r1", "do a task")
            reply = await bus.receive("lead", timeout=2.0)
            assert reply is not None

            # Drain the capture buffer as before_model would.
            flushed = await ext1.before_model(
                state={"messages": state_turn1["messages"]},
                runtime=None,
            )
            assert flushed is not None
            # Two entries: incoming HumanMessage + AIMessage("ok")
            assert len(flushed["messages"]) == 2

            # Simulate add_messages merging into state.
            state_turn1["messages"].extend(flushed["messages"])

            # Turn 1 ends — cleanup.
            await ext1.after_run(state=state_turn1, runtime=None)
            assert ext1._active_team is None
        finally:
            await _shutdown_extension(ext1)

        # State carries tagged messages forward.
        tagged_count = sum(1 for m in state_turn1["messages"] if team_member_of(m) == "r1")
        assert tagged_count == 2

        # --- Turn 2 — fresh extension instance on a different "pod" ---
        ext2 = TeamExtension(agents=[_make_mock_agent("researcher")])
        seen_histories: dict[str, list[Any]] = {}

        def _build(spec: Any, bus: Any) -> Any:
            name = spec["member_name"]

            async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
                seen_histories.setdefault(name, list(state["messages"]))
                return {
                    "messages": list(state["messages"]) + [AIMessage(content="ok2")],
                    "sender": name,
                }

            graph = MagicMock()
            graph.ainvoke = AsyncMock(side_effect=_ainvoke)
            return graph

        ext2.build_teammate_graph = _build  # type: ignore[method-assign]

        try:
            await ext2.rehydrate_if_needed(state_turn1)
            bus2 = ext2._active_team.bus
            await bus2.send("lead", "r1", "continue")
            await bus2.receive("lead", timeout=2.0)

            # r1 should have seen its prior-turn 2 tagged messages + the new one.
            assert len(seen_histories["r1"]) == 3
        finally:
            await _shutdown_extension(ext2)


class TestHookOrderingEnforcement:
    def test_setup_raises_if_history_precedes_team(self):
        from langchain_agentkit.extensions.history.extension import HistoryExtension

        team = TeamExtension(agents=[_make_mock_agent("researcher")])
        history = HistoryExtension(strategy="count", max_messages=50)

        with pytest.raises(ValueError, match="TeamExtension must be listed before"):
            team.setup(extensions=[history, team])

    def test_setup_passes_if_team_precedes_history(self):
        from langchain_agentkit.extensions.history.extension import HistoryExtension

        team = TeamExtension(agents=[_make_mock_agent("researcher")])
        history = HistoryExtension(strategy="count", max_messages=50)

        team.setup(extensions=[team, history])  # Must not raise


class TestDissolveClearsStateThroughReducer:
    """Regression test for the dissolve reducer bug found by team eval.

    The original ``_team_reducer`` was ``right if right is not None
    else left``, which meant ``TeamDissolve`` returning ``{"team": None}``
    silently fell back to ``left`` and the team metadata never cleared.
    LangGraph only invokes the reducer when a node returned an update,
    so seeing ``right=None`` always means "explicit clear" and must
    propagate.  This test exercises the reducer through a real
    ``StateGraph`` to lock in the behavior.
    """

    @pytest.mark.asyncio
    async def test_dissolve_command_clears_team_via_reducer(self):
        from langgraph.graph import END, StateGraph
        from langgraph.types import Command

        from langchain_agentkit.extensions.teams.state import TeamState

        async def _set_team(state: TeamState) -> Command:
            return Command(
                update={
                    "team": {
                        "name": "test-team",
                        "members": [
                            {
                                "member_name": "alice",
                                "kind": "predefined",
                                "agent_id": "researcher",
                            },
                        ],
                    },
                },
            )

        async def _dissolve(state: TeamState) -> Command:
            return Command(update={"team": None})

        workflow: StateGraph = StateGraph(TeamState)
        workflow.add_node("create", _set_team)
        workflow.add_node("dissolve", _dissolve)
        workflow.set_entry_point("create")
        workflow.add_edge("create", "dissolve")
        workflow.add_edge("dissolve", END)

        graph = workflow.compile()
        result = await graph.ainvoke({})

        assert result.get("team") is None, (
            f"Expected dissolve to clear team metadata, got: {result.get('team')}"
        )

    @pytest.mark.asyncio
    async def test_initial_create_persists_team_via_reducer(self):
        """Sanity check: TeamCreate alone (no dissolve) DOES persist."""
        from langgraph.graph import END, StateGraph
        from langgraph.types import Command

        from langchain_agentkit.extensions.teams.state import TeamState

        async def _create(state: TeamState) -> Command:
            return Command(
                update={
                    "team": {
                        "name": "persistent",
                        "members": [],
                    },
                },
            )

        workflow: StateGraph = StateGraph(TeamState)
        workflow.add_node("create", _create)
        workflow.set_entry_point("create")
        workflow.add_edge("create", END)

        result = await workflow.compile().ainvoke({})
        assert result["team"]["name"] == "persistent"
