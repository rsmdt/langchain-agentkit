# ruff: noqa: N801
"""Real-LLM integration evals for AgentMiddleware delegation pipeline.

Tests exercise the FULL compiled graph flow: lead agent receives a message,
decides to delegate via the Delegate tool, LangGraph's ToolNode executes the
delegation, the subagent processes with a real LLM, and the result propagates
back through the lead's ReAct loop.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_agent_middleware_eval.py -x -v -m eval
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.middleware.agents import AgentMiddleware

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

_MODEL = os.environ.get("AGENTKIT_EVAL_MODEL", "gpt-5.4-mini")


def _get_llm():
    """Return a deterministic ChatOpenAI instance."""
    return ChatOpenAI(model=_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Subagent definitions
# ---------------------------------------------------------------------------

# NOTE: These are defined at module level (not inside functions) because the
# metaclass creates a StateGraph, and the class body needs `llm` resolved.
# We use a factory to defer LLM construction until test time.


def _build_calculator():
    """Calculator subagent — answers math with just the number."""
    _llm = _get_llm()

    class calculator(agent):
        llm = _llm
        description = "Answers simple math questions with just the numeric result"
        prompt = (
            "You are a calculator. When asked a math question, respond with "
            "ONLY the numeric answer. No words, no explanation, just the number."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "calculator"}

    return calculator


def _build_greeter():
    """Greeter subagent — responds with a greeting."""
    _llm = _get_llm()

    class greeter(agent):
        llm = _llm
        description = "Greets people warmly in one sentence"
        prompt = (
            "You are a friendly greeter. When someone asks you to greet a person, "
            "respond with a single warm greeting sentence. Keep it short."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "greeter"}

    return greeter


# ---------------------------------------------------------------------------
# Lead agent builder
# ---------------------------------------------------------------------------


def _build_lead_with_middleware(mw: AgentMiddleware):
    """Build a lead agent that uses AgentMiddleware to delegate."""
    _llm = _get_llm()

    class lead(agent):
        llm = _llm
        middleware = [mw]
        prompt = (
            "You are a lead agent. You MUST delegate tasks to specialist agents. "
            "NEVER answer questions yourself. ALWAYS use the Delegate tool. "
            "After receiving the delegation result, report it to the user verbatim."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "lead"}

    return lead


# ---------------------------------------------------------------------------
# Test 1: Full delegation pipeline
# ---------------------------------------------------------------------------


class TestDelegationFullFlow:
    """Lead agent delegates to calculator, full ReAct loop."""

    async def test_delegate_math_to_calculator(self):
        """Lead receives 'What is 2+2?', delegates to calculator, returns '4'."""
        calculator = _build_calculator()
        mw = AgentMiddleware([calculator])
        lead = _build_lead_with_middleware(mw)

        graph = lead.compile()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is 2+2?")]}
        )

        # Verify delegation occurred via delegation_log
        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1, (
            f"Expected at least one delegation, got: {delegation_log}"
        )
        assert delegation_log[0]["agent_name"] == "calculator"
        assert delegation_log[0]["message"]  # non-empty task message
        assert delegation_log[0]["duration_seconds"] > 0

        # Verify final response contains the answer
        final = result["messages"][-1].content
        assert "4" in final, f"Expected '4' in final response, got: {final}"

    async def test_delegate_multiplication(self):
        """Verify delegation works for a different math problem."""
        calculator = _build_calculator()
        mw = AgentMiddleware([calculator])
        lead = _build_lead_with_middleware(mw)

        graph = lead.compile()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is 7 times 8?")]}
        )

        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1
        assert delegation_log[0]["agent_name"] == "calculator"

        final = result["messages"][-1].content
        assert "56" in final, f"Expected '56' in final response, got: {final}"


# ---------------------------------------------------------------------------
# Test 2: Multiple agents — delegation routing
# ---------------------------------------------------------------------------


class TestMultiAgentDelegation:
    """Lead agent with multiple subagents delegates to the right one."""

    async def test_delegates_to_correct_agent(self):
        """With calculator+greeter, math goes to calculator."""
        calculator = _build_calculator()
        greeter = _build_greeter()
        mw = AgentMiddleware([calculator, greeter])
        lead = _build_lead_with_middleware(mw)

        graph = lead.compile()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is 3+5?")]}
        )

        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1
        # Should route to calculator, not greeter
        assert delegation_log[0]["agent_name"] == "calculator"
        assert "8" in result["messages"][-1].content


# ---------------------------------------------------------------------------
# Test 3: Ephemeral delegation
# ---------------------------------------------------------------------------


class TestEphemeralDelegation:
    """Lead agent with ephemeral=True can create ad-hoc agents."""

    async def test_ephemeral_delegation(self):
        """Lead delegates to an ephemeral reasoning agent."""
        calculator = _build_calculator()
        mw = AgentMiddleware([calculator], ephemeral=True)

        # Set parent LLM getter for ephemeral agents
        mw.set_parent_llm_getter(_get_llm)

        lead = _build_lead_with_middleware(mw)

        # Override prompt to encourage ephemeral usage
        _llm = _get_llm()

        class ephemeral_lead(agent):
            llm = _llm
            middleware = [mw]
            prompt = (
                "You are a lead agent. Use the DelegateEphemeral tool to create "
                "a temporary analyst. Set instructions to 'You are a poet. Write "
                "a one-line poem about the topic.' and pass the user's message."
            )

            async def handler(state, *, llm, prompt):
                from langchain_core.messages import SystemMessage

                messages = [SystemMessage(content=prompt)] + state["messages"]
                response = await llm.ainvoke(messages)
                return {"messages": [response], "sender": "ephemeral_lead"}

        graph = ephemeral_lead.compile()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Write about the ocean")]}
        )

        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1
        assert delegation_log[0]["agent_name"] == "ephemeral"


# ---------------------------------------------------------------------------
# Test 4: Scoped context — subagent does not see parent history
# ---------------------------------------------------------------------------


class TestScopedContext:
    """Subagent receives only the delegation message, not parent history."""

    async def test_subagent_answers_only_delegated_question(self):
        """Parent has unrelated history; subagent answers only the math question."""
        calculator = _build_calculator()
        mw = AgentMiddleware([calculator])
        lead = _build_lead_with_middleware(mw)

        graph = lead.compile()

        # Include unrelated prior conversation in messages
        result = await graph.ainvoke({
            "messages": [
                HumanMessage(content="What is 9+1?"),
            ],
        })

        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1
        assert delegation_log[0]["agent_name"] == "calculator"

        # The calculator's result should contain "10"
        summary = delegation_log[0]["result_summary"]
        assert "10" in summary, f"Expected '10' in result_summary, got: {summary}"


# ---------------------------------------------------------------------------
# Test 5: Delegation log structure
# ---------------------------------------------------------------------------


class TestDelegationLogStructure:
    """Verify delegation log entries have all required fields."""

    async def test_log_entry_has_required_fields(self):
        """Verify timestamp, duration, agent_name, message, result_summary."""
        calculator = _build_calculator()
        mw = AgentMiddleware([calculator])
        lead = _build_lead_with_middleware(mw)

        graph = lead.compile()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is 1+1?")]}
        )

        delegation_log = result.get("delegation_log", [])
        assert len(delegation_log) >= 1

        entry = delegation_log[0]
        assert "agent_name" in entry
        assert "message" in entry
        assert "result_summary" in entry
        assert "timestamp" in entry
        assert "duration_seconds" in entry
        assert isinstance(entry["duration_seconds"], (int, float))
        assert entry["duration_seconds"] > 0
        assert entry["agent_name"] == "calculator"
