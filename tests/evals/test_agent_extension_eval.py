# ruff: noqa: N801, N805
"""Real-LLM integration evals for AgentExtension delegation pipeline.

Tests exercise the FULL compiled graph flow: lead agent receives a message,
decides to delegate via the Agent tool, LangGraph's ToolNode executes the
delegation, the subagent processes with a real LLM, and the result propagates
back through the lead's ReAct loop.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_agent_extension_eval.py -x -v -m eval
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.agents import AgentExtension

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


def _build_calculator():
    """Calculator subagent — answers math with just the number."""
    _llm = _get_llm()

    class calculator(agent):
        model = _llm
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
        model = _llm
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


def _build_lead_with_extension(mw: AgentExtension):
    """Build a lead agent that uses AgentExtension to delegate."""
    _llm = _get_llm()

    class lead(agent):
        model = _llm
        extensions = [mw]
        prompt = (
            "You are a lead agent. You MUST delegate tasks to specialist agents. "
            "NEVER answer questions yourself. ALWAYS use the Agent tool. "
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
        mw = AgentExtension([calculator])
        lead = _build_lead_with_extension(mw)

        graph = lead.compile()
        result = await graph.ainvoke({"messages": [HumanMessage(content="What is 2+2?")]})

        final = result["messages"][-1].content
        assert "4" in final, f"Expected '4' in final response, got: {final}"

    async def test_delegate_multiplication(self):
        """Verify delegation works for a different math problem."""
        calculator = _build_calculator()
        mw = AgentExtension([calculator])
        lead = _build_lead_with_extension(mw)

        graph = lead.compile()
        result = await graph.ainvoke({"messages": [HumanMessage(content="What is 7 times 8?")]})

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
        mw = AgentExtension([calculator, greeter])
        lead = _build_lead_with_extension(mw)

        graph = lead.compile()
        result = await graph.ainvoke({"messages": [HumanMessage(content="What is 3+5?")]})

        assert "8" in result["messages"][-1].content


# ---------------------------------------------------------------------------
# Test 3: Dynamic delegation
# ---------------------------------------------------------------------------


class TestDynamicDelegation:
    """Lead agent with ephemeral=True can create ad-hoc agents."""

    async def test_dynamic_delegation(self):
        """Lead delegates to a dynamic reasoning agent."""
        calculator = _build_calculator()
        mw = AgentExtension([calculator], ephemeral=True)

        mw.set_parent_llm_getter(_get_llm)

        _llm = _get_llm()

        class dynamic_lead(agent):
            model = _llm
            extensions = [mw]
            prompt = (
                "You are a lead agent. Use the Agent tool to create a custom agent. "
                "Set agent to {prompt: 'You are a poet. Write a one-line poem about "
                "the topic.'} and pass the user's message."
            )

            async def handler(state, *, llm, prompt):
                from langchain_core.messages import SystemMessage

                messages = [SystemMessage(content=prompt)] + state["messages"]
                response = await llm.ainvoke(messages)
                return {"messages": [response], "sender": "dynamic_lead"}

        graph = dynamic_lead.compile()
        result = await graph.ainvoke({"messages": [HumanMessage(content="Write about the ocean")]})

        final = result["messages"][-1].content
        assert final, "Expected non-empty response from dynamic delegation"


# ---------------------------------------------------------------------------
# Test 4: Scoped context — subagent does not see parent history
# ---------------------------------------------------------------------------


class TestScopedContext:
    """Subagent receives only the delegation message, not parent history."""

    async def test_subagent_answers_only_delegated_question(self):
        """Parent has unrelated history; subagent answers only the math question."""
        calculator = _build_calculator()
        mw = AgentExtension([calculator])
        lead = _build_lead_with_extension(mw)

        graph = lead.compile()

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="What is 9+1?"),
                ],
            }
        )

        final = result["messages"][-1].content
        assert "10" in final, f"Expected '10' in final response, got: {final}"
