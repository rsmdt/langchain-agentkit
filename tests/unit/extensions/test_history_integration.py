# ruff: noqa: N801, N805
"""Integration tests for HistoryExtension through the full graph builder path."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.history import HistoryExtension


def _make_llm(response: AIMessage | None = None) -> MagicMock:
    """Create a mock LLM that returns a canned response."""
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = MagicMock(
        return_value=response or AIMessage(content="response"),
    )
    return mock


class TestHistoryExtensionGraphIntegration:
    """Test HistoryExtension through agent metaclass → compile → ainvoke."""

    @pytest.mark.asyncio
    async def test_handler_receives_truncated_messages(self):
        """The handler (and thus the LLM) sees only the truncated window."""
        received: list[Any] = []

        class truncating_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy="count", max_messages=3)]

            async def handler(state, *, llm):
                received.extend(state["messages"])
                return {
                    "messages": [AIMessage(content="done")],
                    "sender": "truncating_agent",
                }

        compiled = truncating_agent.compile()
        messages = [HumanMessage(content=f"msg-{i}") for i in range(10)]
        await compiled.ainvoke({"messages": messages})

        # Handler should have received only the last 3 messages
        assert len(received) == 3
        assert [m.content for m in received] == ["msg-7", "msg-8", "msg-9"]

    @pytest.mark.asyncio
    async def test_result_contains_truncated_window_plus_response(self):
        """The final result has the kept window + new AI response."""

        class truncating_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy="count", max_messages=2)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="reply")],
                    "sender": "truncating_agent",
                }

        compiled = truncating_agent.compile()
        messages = [HumanMessage(content=f"msg-{i}") for i in range(5)]
        result = await compiled.ainvoke({"messages": messages})

        # Should have kept window (last 2) + response
        contents = [m.content for m in result["messages"]]
        assert contents == ["msg-3", "msg-4", "reply"]

    @pytest.mark.asyncio
    async def test_no_truncation_when_under_limit(self):
        """When messages fit within the limit, all are preserved."""

        class passthrough_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy="count", max_messages=20)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="ok")],
                    "sender": "passthrough_agent",
                }

        compiled = passthrough_agent.compile()
        messages = [HumanMessage(content=f"msg-{i}") for i in range(3)]
        result = await compiled.ainvoke({"messages": messages})

        contents = [m.content for m in result["messages"]]
        assert contents == ["msg-0", "msg-1", "msg-2", "ok"]

    @pytest.mark.asyncio
    async def test_token_strategy_integration(self):
        """Token-based strategy works through the full path."""
        received: list[Any] = []

        def _counter(msg: Any) -> int:
            return len(msg.content)

        class token_agent(agent):
            model = _make_llm()
            extensions = [
                HistoryExtension(
                    strategy="tokens",
                    max_tokens=10,
                    token_counter=_counter,
                )
            ]

            async def handler(state, *, llm):
                received.extend(state["messages"])
                return {
                    "messages": [AIMessage(content="ok")],
                    "sender": "token_agent",
                }

        compiled = token_agent.compile()
        # "aaaa" (4) + "bbbb" (4) + "cccc" (4) = 12 > 10
        messages = [
            HumanMessage(content="aaaa"),
            HumanMessage(content="bbbb"),
            HumanMessage(content="cccc"),
        ]
        await compiled.ainvoke({"messages": messages})

        # Should have truncated to fit within 10 tokens
        assert len(received) == 2
        assert [m.content for m in received] == ["bbbb", "cccc"]

    @pytest.mark.asyncio
    async def test_custom_strategy_integration(self):
        """Custom strategy object works through the full path."""

        class KeepLast:
            def transform(self, messages: list[Any]) -> list[Any]:
                return messages[-1:] if messages else []

        class custom_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy=KeepLast())]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="done")],
                    "sender": "custom_agent",
                }

        compiled = custom_agent.compile()
        messages = [HumanMessage(content=f"msg-{i}") for i in range(10)]
        result = await compiled.ainvoke({"messages": messages})

        contents = [m.content for m in result["messages"]]
        assert contents == ["msg-9", "done"]


class TestHistoryExtensionCheckpointerPersistence:
    """Verify that truncation is persisted in the checkpointer — not just
    applied to the LLM input. This is the load-bearing claim in
    HistoryExtension's docstring: "the checkpointer stays lean"."""

    @pytest.mark.asyncio
    async def test_checkpointer_state_is_truncated(self):
        """After one turn, the checkpointed messages list is the kept
        window + response — dropped messages are gone from state."""
        from langgraph.checkpoint.memory import InMemorySaver

        class truncating_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy="count", max_messages=3)]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="reply-1")],
                    "sender": "truncating_agent",
                }

        compiled = truncating_agent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}
        messages = [HumanMessage(content=f"msg-{i}") for i in range(10)]
        await compiled.ainvoke({"messages": messages}, config=config)

        persisted = compiled.get_state(config).values["messages"]
        contents = [m.content for m in persisted]

        # Only last 3 + response survive in the checkpointer
        assert contents == ["msg-7", "msg-8", "msg-9", "reply-1"]
        assert "msg-0" not in contents
        assert "msg-6" not in contents

    @pytest.mark.asyncio
    async def test_truncation_holds_across_two_turns_on_same_thread(self):
        """Resuming the same thread: the second turn's LLM input is
        truncated from the persisted (already-lean) prefix + new input,
        and the final checkpointed state remains within the window."""
        from langgraph.checkpoint.memory import InMemorySaver

        received_per_turn: list[list[Any]] = []
        responses = iter([AIMessage(content="reply-1"), AIMessage(content="reply-2")])

        class truncating_agent(agent):
            model = _make_llm()
            extensions = [HistoryExtension(strategy="count", max_messages=3)]

            async def handler(state, *, llm):
                received_per_turn.append(list(state["messages"]))
                return {
                    "messages": [next(responses)],
                    "sender": "truncating_agent",
                }

        compiled = truncating_agent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}

        # Turn 1: 5 messages in, window=3 → handler sees last 3
        turn1 = [HumanMessage(content=f"a-{i}") for i in range(5)]
        await compiled.ainvoke({"messages": turn1}, config=config)

        after_turn1 = [m.content for m in compiled.get_state(config).values["messages"]]
        assert after_turn1 == ["a-2", "a-3", "a-4", "reply-1"]

        # Turn 2: append 2 new messages to the same thread
        turn2 = [HumanMessage(content="b-0"), HumanMessage(content="b-1")]
        await compiled.ainvoke({"messages": turn2}, config=config)

        # Handler on turn 2 should see last 3 of (persisted-lean + new) =
        # ["a-2","a-3","a-4","reply-1","b-0","b-1"] → ["reply-1","b-0","b-1"]
        turn2_seen = [m.content for m in received_per_turn[1]]
        assert turn2_seen == ["reply-1", "b-0", "b-1"]

        # And the checkpointer still holds only kept window + reply-2
        after_turn2 = [m.content for m in compiled.get_state(config).values["messages"]]
        assert after_turn2 == ["reply-1", "b-0", "b-1", "reply-2"]
        # First-turn human inputs must be gone — they were dropped in turn 1
        # and never resurrected by the resumption path.
        assert "a-0" not in after_turn2
        assert "a-4" not in after_turn2
