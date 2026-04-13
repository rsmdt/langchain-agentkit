# ruff: noqa: N801, N805
"""Integration tests for MessagePersistenceExtension through the full graph."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.history import HistoryExtension
from langchain_agentkit.extensions.persistence import MessagePersistenceExtension


class FakeSink:
    """In-memory async callback that records each persist invocation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_on_next = False

    async def __call__(self, *, thread_id: str | None, messages: list[Any]) -> None:
        if self.raise_on_next:
            self.raise_on_next = False
            raise RuntimeError("sink boom")
        self.calls.append({"thread_id": thread_id, "messages": list(messages)})


def _make_llm() -> MagicMock:
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = MagicMock(return_value=AIMessage(content="unused"))
    return mock


class TestMessagePersistenceSingleTurn:
    @pytest.mark.asyncio
    async def test_persists_only_newly_generated_messages(self):
        """Turn delta = messages whose IDs were not present at before_run."""
        sink = FakeSink()

        class one_shot(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="hello back", id="ai-1")],
                    "sender": "one_shot",
                }

        compiled = one_shot.compile()
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi", id="h-1")]},
            config={"configurable": {"thread_id": "t-single"}},
        )

        assert len(sink.calls) == 1
        call = sink.calls[0]
        assert call["thread_id"] == "t-single"
        # Only the AI reply is persisted — the user input (h-1) was
        # already in state when before_run fired.
        contents = [m.content for m in call["messages"]]
        assert contents == ["hello back"]

    @pytest.mark.asyncio
    async def test_no_call_when_no_new_messages(self):
        """If the handler returns no new messages, the sink is not called."""
        sink = FakeSink()

        class silent(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                return {"messages": [], "sender": "silent"}

        compiled = silent.compile()
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi", id="h-1")]},
            config={"configurable": {"thread_id": "t-silent"}},
        )

        assert sink.calls == []

    @pytest.mark.asyncio
    async def test_messages_without_explicit_ids_are_still_captured(self):
        """LangGraph's add_messages reducer auto-assigns UUID IDs.

        This means the ID-based snapshot/diff works even when callers
        don't set ``id`` explicitly — the reducer guarantees every
        message in state has an ID by the time hooks observe state.
        """
        sink = FakeSink()

        class unidentified(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                # No explicit id — the reducer will assign one.
                return {"messages": [AIMessage(content="x")], "sender": "unidentified"}

        compiled = unidentified.compile()
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": "t-noid"}},
        )

        assert len(sink.calls) == 1
        contents = [m.content for m in sink.calls[0]["messages"]]
        assert contents == ["x"]
        # Every captured message has an ID assigned by the reducer.
        assert all(getattr(m, "id", None) for m in sink.calls[0]["messages"])

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_fail_turn(self):
        """Exceptions in the callback are logged, not propagated."""
        sink = FakeSink()
        sink.raise_on_next = True

        class robust(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="ok", id="ai-1")],
                    "sender": "robust",
                }

        compiled = robust.compile()
        # Should not raise.
        result = await compiled.ainvoke(
            {"messages": [HumanMessage(content="hi", id="h-1")]},
            config={"configurable": {"thread_id": "t-fail"}},
        )
        assert result["messages"][-1].content == "ok"

    @pytest.mark.asyncio
    async def test_thread_id_absent_still_persists(self):
        """Missing thread_id is forwarded as None — sink decides what to do."""
        sink = FakeSink()

        class no_thread(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="ok", id="ai-1")],
                    "sender": "no_thread",
                }

        compiled = no_thread.compile()
        await compiled.ainvoke({"messages": [HumanMessage(content="hi", id="h-1")]})

        assert len(sink.calls) == 1
        assert sink.calls[0]["thread_id"] is None


class TestMessagePersistenceMultiTurn:
    @pytest.mark.asyncio
    async def test_two_turns_same_thread_each_persists_its_own_delta(self):
        """Each ainvoke persists only what was generated in that invocation."""
        from langgraph.checkpoint.memory import InMemorySaver

        sink = FakeSink()
        responses = iter(
            [
                AIMessage(content="reply-1", id="ai-1"),
                AIMessage(content="reply-2", id="ai-2"),
            ]
        )

        class conv(agent):
            model = _make_llm()
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm):
                return {"messages": [next(responses)], "sender": "conv"}

        compiled = conv.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t-multi"}}

        await compiled.ainvoke({"messages": [HumanMessage(content="u1", id="h-1")]}, config=config)
        await compiled.ainvoke({"messages": [HumanMessage(content="u2", id="h-2")]}, config=config)

        assert len(sink.calls) == 2

        turn1_contents = [m.content for m in sink.calls[0]["messages"]]
        turn2_contents = [m.content for m in sink.calls[1]["messages"]]

        # Turn 1: only reply-1. The h-1 HumanMessage was already in state
        # at before_run (input was applied by reducer before _run_entry).
        assert turn1_contents == ["reply-1"]
        # Turn 2: only reply-2. h-1, ai-1 were already seen; h-2 also was
        # in state at before_run of turn 2 (reducer applied the input).
        assert turn2_contents == ["reply-2"]


class TestMessagePersistenceWithHistoryExtension:
    @pytest.mark.asyncio
    async def test_id_based_delta_survives_truncation(self):
        """HistoryExtension truncates state but our ID snapshot only shrinks.

        This proves the ID-set approach is robust: even if truncation
        drops pre-turn messages from state between before_run and
        after_run, we never misidentify them as "generated this turn".
        """
        sink = FakeSink()

        class truncating(agent):
            model = _make_llm()
            # Window=2 with lots of prior messages — truncation will drop
            # pre-turn messages between before_run and after_run.
            extensions = [
                HistoryExtension(strategy="count", max_messages=2),
                MessagePersistenceExtension(persist=sink),
            ]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="new-reply", id="ai-new")],
                    "sender": "truncating",
                }

        compiled = truncating.compile()
        prior = [HumanMessage(content=f"msg-{i}", id=f"h-{i}") for i in range(10)]
        await compiled.ainvoke(
            {"messages": prior},
            config={"configurable": {"thread_id": "t-trunc"}},
        )

        assert len(sink.calls) == 1
        contents = [m.content for m in sink.calls[0]["messages"]]
        # Only the genuinely new AI message — none of the truncated-out
        # human messages should be reported as "generated this turn".
        assert contents == ["new-reply"]


class TestMessagePersistenceToolMessages:
    @pytest.mark.asyncio
    async def test_tool_messages_are_captured_in_turn_delta(self):
        """A turn spanning model→tool→model persists all new AI and tool messages."""
        from langchain_core.tools import tool

        sink = FakeSink()

        @tool
        def echo(text: str) -> str:
            """Echo back the input."""
            return f"echoed: {text}"

        # Handler alternates: first call emits a tool_call, second emits final reply.
        call_count = {"n": 0}

        class with_tools(agent):
            model = _make_llm()
            tools = [echo]
            extensions = [MessagePersistenceExtension(persist=sink)]

            async def handler(state, *, llm, tools):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    ai = AIMessage(
                        content="",
                        id="ai-tool",
                        tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "tc-1"}],
                    )
                    return {"messages": [ai], "sender": "with_tools"}
                return {
                    "messages": [AIMessage(content="done", id="ai-final")],
                    "sender": "with_tools",
                }

        compiled = with_tools.compile()
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="go", id="h-1")]},
            config={"configurable": {"thread_id": "t-tools"}},
        )

        assert len(sink.calls) == 1
        persisted = sink.calls[0]["messages"]
        # Should include: initial AI with tool_call, the ToolMessage from ToolNode,
        # and the final AI reply. User input (h-1) is excluded.
        kinds = [type(m).__name__ for m in persisted]
        assert "AIMessage" in kinds
        assert "ToolMessage" in kinds
        assert kinds.count("AIMessage") == 2
        # Verify the tool message content
        tool_msgs = [m for m in persisted if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "echoed: hi" in str(tool_msgs[0].content)
        # No HumanMessage should be persisted (it was seen at turn start)
        assert "HumanMessage" not in kinds
