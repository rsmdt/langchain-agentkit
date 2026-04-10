# ruff: noqa: N801, N805
"""Tests for ``_graph_builder`` helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_agentkit._graph_builder import build_ephemeral_graph


class _EchoInput(BaseModel):
    value: str = Field(description="value to echo back")


def _make_echo_tool() -> StructuredTool:
    async def _echo(value: str) -> str:
        return f"echoed:{value}"

    return StructuredTool.from_function(
        coroutine=_echo,
        name="echo",
        description="Echo a value.",
        args_schema=_EchoInput,
    )


class TestBuildEphemeralGraphBindsTools:
    """``build_ephemeral_graph`` must bind user_tools so the LLM can emit tool calls.

    Historically the inner handler called ``llm.ainvoke(msgs)`` without
    ``bind_tools``, which meant ephemeral agents (including team dynamic
    members and their proxy task tools) could never emit tool calls.  The
    framework contract is "handlers bind their own tools" — the ephemeral
    handler is itself a handler, so it must honor that contract.
    """

    @pytest.mark.asyncio
    async def test_bind_tools_called_with_user_tools(self):
        """The inner handler binds user_tools before invoking the LLM."""
        echo = _make_echo_tool()

        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        graph = build_ephemeral_graph(
            name="tester",
            llm=mock_llm,
            prompt="you are a tester",
            user_tools=[echo],
        )

        await graph.ainvoke({"messages": [HumanMessage(content="go")]})

        # bind_tools must have been called with the user_tools we passed.
        mock_llm.bind_tools.assert_called_once()
        called_tools = mock_llm.bind_tools.call_args[0][0]
        assert len(called_tools) == 1
        assert called_tools[0].name == "echo"

        # And the bound LLM is what actually got invoked.
        bound_llm.ainvoke.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_tools_skips_bind(self):
        """When no tools are provided, bind_tools must NOT be called."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        mock_llm.bind_tools = MagicMock()

        graph = build_ephemeral_graph(
            name="tester",
            llm=mock_llm,
            prompt="you are a tester",
        )

        await graph.ainvoke({"messages": [HumanMessage(content="go")]})

        mock_llm.bind_tools.assert_not_called()
        mock_llm.ainvoke.assert_awaited()

    @pytest.mark.asyncio
    async def test_tool_call_round_trip(self):
        """End-to-end: bound LLM emits a tool call, ToolNode runs it, loop continues."""
        echo = _make_echo_tool()

        # First invocation: LLM emits a tool call.
        # Second invocation: LLM returns a final answer.
        call_count = {"n": 0}

        async def _fake_ainvoke(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "t1",
                            "name": "echo",
                            "args": {"value": "hello"},
                        }
                    ],
                )
            return AIMessage(content="final")

        bound_llm = MagicMock()
        bound_llm.ainvoke = AsyncMock(side_effect=_fake_ainvoke)

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=bound_llm)

        graph = build_ephemeral_graph(
            name="tester",
            llm=mock_llm,
            prompt="you are a tester",
            user_tools=[echo],
        )

        result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

        # The tool must have actually run — its result should be in the
        # message list.
        tool_results = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
        assert len(tool_results) == 1
        assert "echoed:hello" in tool_results[0].content
        # And the final AI response comes after the tool result.
        assert result["messages"][-1].content == "final"
