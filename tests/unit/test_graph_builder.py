# ruff: noqa: N801, N805
"""Tests for ``graph_builder`` helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_agentkit._internal.graph_builder import (
    _apply_after_update,
    _post_handler_state,
    build_ephemeral_graph,
)


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


class TestApplyAfterUpdate:
    """``_apply_after_update`` lifts jump_to without dropping sibling keys.

    A self-correction hook (e.g. the rubric grader) must inject a message AND
    route back to the model in one update. Historically a jump_to update had
    its other keys discarded, so the injected feedback never reached the model.
    """

    def test_jump_to_preserves_injected_message_and_bookkeeping(self):
        result: dict[str, Any] = {"messages": [AIMessage(content="draft")]}
        revision = HumanMessage(content="fix it")
        _apply_after_update(
            result,
            {"jump_to": "model", "messages": [revision], "_rubric_iterations": 1},
        )
        assert result["_agentkit_jump_to"] == "model"
        assert result["_rubric_iterations"] == 1
        # Injected message lands AFTER the model's output (add_messages order).
        assert result["messages"] == [result["messages"][0], revision]
        assert result["messages"][-1] is revision

    def test_jump_to_without_messages_still_merges_keys(self):
        result: dict[str, Any] = {"messages": [AIMessage(content="x")]}
        _apply_after_update(result, {"jump_to": "end", "_turn_budget_used": 1})
        assert result["_agentkit_jump_to"] == "end"
        assert result["_turn_budget_used"] == 1
        assert len(result["messages"]) == 1  # unchanged

    def test_non_jump_update_merges_normally(self):
        result: dict[str, Any] = {"messages": [AIMessage(content="x")]}
        _apply_after_update(result, {"_some_key": "v"})
        assert result["_some_key"] == "v"
        assert "_agentkit_jump_to" not in result


class TestPostHandlerState:
    """``_post_handler_state`` exposes the model's just-produced output to hooks."""

    def test_messages_concatenated_and_keys_merged(self):
        state = {"messages": [HumanMessage(content="go")], "rubric": "- r"}
        new_ai = AIMessage(content="answer")
        post = _post_handler_state(state, {"messages": [new_ai], "sender": "agent"})
        assert post["messages"] == [state["messages"][0], new_ai]
        assert post["sender"] == "agent"
        assert post["rubric"] == "- r"

    def test_does_not_mutate_inputs(self):
        state = {"messages": [HumanMessage(content="go")]}
        result = {"messages": [AIMessage(content="answer")]}
        _post_handler_state(state, result)
        assert len(state["messages"]) == 1
        assert len(result["messages"]) == 1

    def test_non_dict_result_returns_state(self):
        state = {"messages": []}
        assert _post_handler_state(state, "not-a-dict") is state


class TestBuildEphemeralGraphMaxTurns:
    """``max_turns`` must be applied as ``recursion_limit`` on the compiled graph."""

    @pytest.mark.asyncio
    async def test_max_turns_sets_recursion_limit(self):
        """max_turns=N → compiled.config['recursion_limit'] == N * 2."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))

        graph = build_ephemeral_graph(
            name="bounded",
            llm=mock_llm,
            prompt="you are bounded",
            max_turns=6,
        )

        assert graph.config["recursion_limit"] == 12

    @pytest.mark.asyncio
    async def test_no_max_turns_no_recursion_limit(self):
        """Absent max_turns leaves recursion_limit unset (LangGraph default applies)."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))

        graph = build_ephemeral_graph(
            name="unbounded",
            llm=mock_llm,
            prompt="you are unbounded",
        )

        assert "recursion_limit" not in (getattr(graph, "config", None) or {})
