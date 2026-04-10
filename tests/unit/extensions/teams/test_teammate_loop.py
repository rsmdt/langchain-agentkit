"""Tests for ``_teammate_loop`` — capture buffer, tagging, error handling."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.extensions.teams.bus import TeamMessageBus, _teammate_loop
from langchain_agentkit.extensions.teams.filter import TEAM_KEY, is_team_tagged


def _make_graph_returning(new_messages_per_call: list[list[Any]]) -> Any:
    """Build a mock compiled graph that returns full histories.

    The fake graph receives ``{"messages": history, ...}`` and must
    return ``{"messages": history + new}`` (matching ``add_messages``
    semantics).  The ``new_messages_per_call`` list provides the
    messages to append on each successive invocation.
    """
    calls = {"n": 0}

    async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
        incoming = list(state.get("messages") or [])
        new = list(new_messages_per_call[calls["n"]])
        calls["n"] += 1
        return {"messages": incoming + new, "sender": state.get("sender", "")}

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=_ainvoke)
    return graph


class TestTeammateLoopBasics:
    @pytest.mark.asyncio
    async def test_captures_and_tags_new_messages(self):
        """Incoming bus message and produced AI reply both land tagged in buffer."""
        bus = TeamMessageBus()
        bus.register("r1")
        bus.register("lead")

        capture: list[Any] = []
        graph = _make_graph_returning(
            [
                [AIMessage(content="researched X")],
            ],
        )

        # Start the loop
        loop_task = asyncio.create_task(
            _teammate_loop(
                "r1",
                graph,
                bus,
                initial_history=[],
                capture_buffer=capture,
            ),
        )

        # Send one instruction and one shutdown
        await bus.send("lead", "r1", "research X")
        reply = await bus.receive("lead", timeout=2.0)
        assert reply is not None
        assert reply.content == "researched X"

        await bus.send("lead", "r1", '{"type":"shutdown_request"}')
        await asyncio.wait_for(loop_task, timeout=2.0)

        # Capture buffer should contain the incoming HumanMessage and
        # the teammate's AIMessage, both tagged.
        assert len(capture) == 2
        human, ai = capture
        assert isinstance(human, HumanMessage)
        assert human.content == "research X"
        assert is_team_tagged(human)
        assert human.additional_kwargs[TEAM_KEY]["member"] == "r1"

        assert isinstance(ai, AIMessage)
        assert ai.content == "researched X"
        assert is_team_tagged(ai)
        assert ai.name == "r1"

    @pytest.mark.asyncio
    async def test_initial_history_seeds_local_state(self):
        """Prior-turn history is passed to ainvoke on the first message."""
        bus = TeamMessageBus()
        bus.register("r1")
        bus.register("lead")

        prior_human = HumanMessage(
            content="earlier instruction",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        prior_ai = AIMessage(
            content="earlier reply",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
            name="r1",
        )
        initial = [prior_human, prior_ai]

        # Snapshot the messages list at invocation time — the loop
        # extends the same list after ainvoke returns, so we can't
        # rely on MagicMock's stored reference.
        seen_at_call: list[list[Any]] = []

        async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
            seen_at_call.append(list(state["messages"]))
            return {
                "messages": list(state["messages"]) + [AIMessage(content="new reply")],
                "sender": state.get("sender", ""),
            }

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=_ainvoke)

        capture: list[Any] = []
        loop_task = asyncio.create_task(
            _teammate_loop(
                "r1",
                graph,
                bus,
                initial_history=initial,
                capture_buffer=capture,
            ),
        )

        await bus.send("lead", "r1", "continue")
        reply = await bus.receive("lead", timeout=2.0)
        assert reply is not None
        assert reply.content == "new reply"

        await bus.send("lead", "r1", '{"type":"shutdown_request"}')
        await asyncio.wait_for(loop_task, timeout=2.0)

        # At invocation time, graph saw initial (2) + new human (1) = 3 msgs.
        assert len(seen_at_call) == 1
        invoked_msgs = seen_at_call[0]
        assert len(invoked_msgs) == 3
        assert invoked_msgs[0] is prior_human
        assert invoked_msgs[1] is prior_ai
        assert invoked_msgs[2].content == "continue"

        # Buffer only has what was produced THIS turn (the loop doesn't
        # re-capture initial_history).
        assert len(capture) == 2  # incoming + reply

    @pytest.mark.asyncio
    async def test_exception_captured_as_tagged_error(self):
        """Graph failures become a tagged AIMessage in the capture buffer."""
        bus = TeamMessageBus()
        bus.register("r1")
        bus.register("lead")

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        capture: list[Any] = []
        loop_task = asyncio.create_task(
            _teammate_loop(
                "r1",
                graph,
                bus,
                initial_history=[],
                capture_buffer=capture,
            ),
        )

        await bus.send("lead", "r1", "do work")
        reply = await bus.receive("lead", timeout=2.0)
        assert reply is not None
        assert "Error during execution" in reply.content

        await bus.send("lead", "r1", '{"type":"shutdown_request"}')
        await asyncio.wait_for(loop_task, timeout=2.0)

        # Buffer contains the incoming human + the error AI message.
        assert len(capture) == 2
        error_msg = capture[1]
        assert isinstance(error_msg, AIMessage)
        assert "boom" in error_msg.content
        assert is_team_tagged(error_msg)

    @pytest.mark.asyncio
    async def test_non_append_reducer_raises_assertion(self):
        """If the graph returns fewer messages than input, assertion fires."""
        bus = TeamMessageBus()
        bus.register("r1")
        bus.register("lead")

        # Graph that returns an EMPTY messages list regardless of input.
        async def _broken_ainvoke(state: dict[str, Any]) -> dict[str, Any]:
            return {"messages": [], "sender": "r1"}

        graph = MagicMock()
        graph.ainvoke = AsyncMock(side_effect=_broken_ainvoke)

        capture: list[Any] = []
        loop_task = asyncio.create_task(
            _teammate_loop(
                "r1",
                graph,
                bus,
                initial_history=[],
                capture_buffer=capture,
            ),
        )

        await bus.send("lead", "r1", "do work")

        with pytest.raises(AssertionError, match="non-append reducer"):
            await asyncio.wait_for(loop_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_tool_messages_captured_with_team_tag(self):
        """ToolMessage tool results in the teammate's loop get tagged."""
        bus = TeamMessageBus()
        bus.register("r1")
        bus.register("lead")

        tool_result = ToolMessage(
            content='{"result":"ok"}',
            tool_call_id="t1",
            name="TaskCreate",  # tool name, NOT teammate name
        )
        ai_final = AIMessage(content="task created")
        graph = _make_graph_returning([[tool_result, ai_final]])

        capture: list[Any] = []
        loop_task = asyncio.create_task(
            _teammate_loop(
                "r1",
                graph,
                bus,
                initial_history=[],
                capture_buffer=capture,
            ),
        )

        await bus.send("lead", "r1", "create a task")
        await bus.receive("lead", timeout=2.0)
        await bus.send("lead", "r1", '{"type":"shutdown_request"}')
        await asyncio.wait_for(loop_task, timeout=2.0)

        # Capture: incoming human, tool result, final ai — all tagged.
        assert len(capture) == 3
        assert all(is_team_tagged(m) for m in capture)
        # ToolMessage keeps its .name as the tool name but has the team tag.
        assert capture[1].name == "TaskCreate"
        assert capture[1].additional_kwargs[TEAM_KEY]["member"] == "r1"
