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
