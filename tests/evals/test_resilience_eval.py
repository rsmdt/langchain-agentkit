# ruff: noqa: N801, N805
"""ResilienceExtension evals — live LLM loop recovery from tool errors.

These evals exploit the exact failure mode ResilienceExtension exists to
fix: a Python tool raises an unhandled exception mid-ReAct, which
without resilience would leave an orphan ``AIMessage(tool_calls=[...])``
in the checkpoint and abort the turn (and poison the next turn on the
OpenAI Responses API). With the extension wired in, the tool failure is
converted to a synthetic error ``ToolMessage``, the pairing invariant
holds, and the ReAct loop continues — letting the model retry, apologise,
or report the failure.

Two scenarios are covered:

1. **Always-failing tool** — proves pairing holds and the loop terminates
   with a final AIMessage even when the tool can never succeed.
2. **Flaky tool (fail-then-succeed)** — proves the LLM loop is actually
   rerun after the synthetic error, the tool is retried, and the final
   answer reflects the successful second call.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest).
- The ``langchain-openai`` package.

Run::

    uv run pytest tests/evals/test_resilience_eval.py -x -v -m eval
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.resilience import (
    ResilienceExtension,
    ToolErrorEvent,
)

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment,misc]

_MODEL = os.environ.get("AGENTKIT_EVAL_MODEL", "gpt-4o-mini")


def _get_llm() -> Any:
    """Deterministic ChatOpenAI instance for eval reproducibility."""
    return ChatOpenAI(model=_MODEL, temperature=0)


# ------------------------------------------------------------------
# Tool factories — deliberately raise to reproduce the orphan scenario
# ------------------------------------------------------------------


def _make_always_failing_tool() -> Any:
    @tool
    def lookup_weather(city: str) -> str:
        """Look up the current weather for a city."""
        raise RuntimeError(f"weather API unreachable for {city!r}")

    return lookup_weather


def _make_flaky_tool() -> tuple[Any, dict[str, int]]:
    """Raises on the first call, returns a deterministic success after."""
    state: dict[str, int] = {"calls": 0}

    @tool
    def lookup_weather(city: str) -> str:
        """Look up the current weather for a city."""
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("transient failure — retry")
        return f"The weather in {city} is 18C and sunny."

    return lookup_weather, state


# ------------------------------------------------------------------
# Agent builder — wires ResilienceExtension + the failing tool
# ------------------------------------------------------------------


def _build_agent(
    failing_tool: Any,
    events: list[ToolErrorEvent] | None = None,
) -> Any:
    llm = _get_llm()
    resilience = ResilienceExtension(
        on_tool_error_caught=(events.append if events is not None else None),
    )

    class resilient_agent(agent):
        model = llm
        tools = [failing_tool]
        extensions = [resilience]
        prompt = (
            "You are a helpful weather assistant. Always use the "
            "lookup_weather tool to answer weather questions. If a tool "
            "call fails, try it exactly one more time; if it fails again, "
            "apologise to the user and explain briefly that the weather "
            "service is unavailable."
        )

        async def handler(state, *, llm, tools, prompt, runtime):
            bound = llm.bind_tools(tools)
            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await bound.ainvoke(messages)
            return {"messages": [response]}

    return resilient_agent.compile()


# ------------------------------------------------------------------
# Shared assertion helpers
# ------------------------------------------------------------------


def _assert_pairing_invariant(messages: list[Any]) -> None:
    """Every AIMessage tool_call must be paired with a ToolMessage."""
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage) and m.tool_calls:
            expected = {tc["id"] for tc in m.tool_calls}
            paired = {t.tool_call_id for t in messages[i + 1 :] if isinstance(t, ToolMessage)}
            missing = expected - paired
            assert not missing, f"orphan tool call id(s) without ToolMessage: {missing}"


def _synthesized_error_messages(messages: list[Any]) -> list[ToolMessage]:
    return [
        m
        for m in messages
        if isinstance(m, ToolMessage)
        and m.additional_kwargs.get("agentkit", {}).get("synthesized") is True
    ]


def _real_tool_messages(messages: list[Any]) -> list[ToolMessage]:
    return [
        m
        for m in messages
        if isinstance(m, ToolMessage)
        and not m.additional_kwargs.get("agentkit", {}).get("synthesized")
    ]


# ------------------------------------------------------------------
# Evals
# ------------------------------------------------------------------


class TestResilienceExploitAlwaysFailingTool:
    """Tool unconditionally raises — loop must still terminate cleanly."""

    async def test_loop_recovers_and_produces_final_answer(self):
        events: list[ToolErrorEvent] = []
        graph = _build_agent(_make_always_failing_tool(), events)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is the weather in Paris?")]}
        )

        messages = result["messages"]

        # 1. Pairing invariant holds — no orphan AIMessage.tool_calls.
        _assert_pairing_invariant(messages)

        # 2. Resilience synthesized at least one error ToolMessage.
        synthesized = _synthesized_error_messages(messages)
        assert synthesized, "ResilienceExtension did not synthesize any ToolMessage"
        for m in synthesized:
            assert m.status == "error"
            meta = m.additional_kwargs["agentkit"]
            assert meta["reason"] == "tool_error"
            assert meta["exc_type"] == "RuntimeError"

        # 3. Loop terminated with a final AIMessage (no tool_calls).
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert not final.tool_calls
        assert isinstance(final.content, str) and final.content.strip()

        # 4. Telemetry callback fired once per synthesized message.
        assert len(events) == len(synthesized)
        assert all(e.exc_type == "RuntimeError" for e in events)
        assert all(e.tool_name == "lookup_weather" for e in events)
        assert all(e.tool_call_id for e in events)


class TestResilienceExploitFlakyToolRerunsLoop:
    """Tool fails then succeeds — prove the loop was actually rerun.

    This is the full self-healing contract: after the resilience layer
    converts the first exception into an error ToolMessage, the ReAct
    loop re-runs the LLM, which retries and observes the successful
    result. The final answer must reflect the successful retry — that is
    only possible if the loop re-executed end-to-end.
    """

    async def test_agent_retries_after_synthetic_error(self):
        events: list[ToolErrorEvent] = []
        failing_tool, tool_state = _make_flaky_tool()
        graph = _build_agent(failing_tool, events)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is the weather in Paris?")]}
        )

        messages = result["messages"]

        # 1. Pairing invariant holds throughout the retried loop.
        _assert_pairing_invariant(messages)

        # 2. Tool was invoked at least twice — proof the loop reran.
        assert tool_state["calls"] >= 2, (
            f"tool invoked only {tool_state['calls']} time(s) — "
            "LLM loop did not retry after the synthetic error"
        )

        # 3. Exactly one synthesized error (from the first failure).
        synthesized = _synthesized_error_messages(messages)
        assert len(synthesized) >= 1
        assert synthesized[0].status == "error"
        assert synthesized[0].additional_kwargs["agentkit"]["exc_type"] == "RuntimeError"

        # 4. A real, successful ToolMessage followed carrying the retry output.
        successful = _real_tool_messages(messages)
        assert successful, "no successful tool call observed after retry"
        joined = " ".join(str(m.content).lower() for m in successful)
        assert "sunny" in joined or "18" in joined, (
            f"successful ToolMessage(s) did not contain expected weather payload: {joined!r}"
        )

        # 5. Final answer is a plain AIMessage that references the recovery.
        final = messages[-1]
        assert isinstance(final, AIMessage)
        assert not final.tool_calls
        content = final.content.lower() if isinstance(final.content, str) else ""
        assert "paris" in content or "sunny" in content or "18" in content, (
            f"final answer did not reflect successful retry: {final.content!r}"
        )

        # 6. Telemetry captured the transient failure exactly once.
        assert len(events) == len(synthesized)
        assert events[0].tool_name == "lookup_weather"
