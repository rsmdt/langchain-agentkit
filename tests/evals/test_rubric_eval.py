# ruff: noqa: N801, N805
"""RubricExtension evals — live LLM rubric-gated iteration in both modes.

These evals exercise the exact behavior the extension exists to provide, using
*forcing functions* so a real LLM reliably engages the loop rather than passing
on the first try:

1. **Auto mode** — the rubric demands a marker the agent has no way to know
   until the grader's feedback reveals it, so the first attempt MUST fail and
   the autonomous revision loop MUST run for the marker to appear. Proves the
   closed loop re-drafts on the grader's feedback until ``satisfied``.

2. **User mode** — the rubric requires a fact only the user can supply, so the
   agent MUST yield and ask, then satisfy once the user answers. Proves the
   open loop hands control back to the user across turns and converges.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest).
- The ``langchain-openai`` package.

Run::

    uv run pytest tests/evals/test_rubric_eval.py -x -v -m eval
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_agentkit import Agent
from langchain_agentkit.extensions.rubric import RubricEvaluation, RubricExtension
from tests.evals.conftest import EVAL_MODEL

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


def _llm() -> Any:
    return ChatOpenAI(model=EVAL_MODEL, temperature=0)


@tool
def scratch(note: str) -> str:
    """Jot a private working note. Has no side effects."""
    return "noted"


async def _agentkit_handler(state: Any, *, llm: Any, tools: Any, prompt: Any) -> dict[str, Any]:
    bound = llm.bind_tools(tools) if tools else llm
    messages = ([SystemMessage(content=prompt)] if prompt else []) + state["messages"]
    return {"messages": [await bound.ainvoke(messages)]}


# ------------------------------------------------------------------
# Auto mode — the closed loop must re-draft to discover the marker
# ------------------------------------------------------------------


class TestRubricAutoModeLoop:
    async def test_loop_revises_until_marker_present(self):
        evals: list[RubricEvaluation] = []
        grader = RubricExtension(
            model=_llm(),
            mode="auto",
            review={"stop": 4},
            on_evaluation=evals.append,
        )

        class AutoAgent(Agent):
            model = _llm()
            tools = [scratch]  # gives the graph a tool so jump_to is honored
            extensions = [grader]
            prompt = "You are a concise assistant. Answer the user directly."

            handler = staticmethod(_agentkit_handler)

        graph = await AutoAgent().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "rubric-auto"}}

        # The agent is NOT told the marker requirement — only the grader's
        # rubric knows it, so the first draft cannot satisfy and the loop must
        # run for the marker to appear.
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="In one sentence, what is the capital of France?")
                ],
                "rubric": (
                    "- The answer states that the capital of France is Paris\n"
                    "- The response ends with this exact marker on its own line: DONE-7Q2"
                ),
            },
            config=config,
        )
        state = (await graph.aget_state(config)).values

        # 1. Converged to satisfied.
        assert state["_rubric_status"] == "satisfied", (
            f"expected satisfied, got {state['_rubric_status']} "
            f"after {state['_rubric_iterations']} reviews"
        )

        # 2. The loop actually re-drafted — at least one needs_revision before
        #    satisfied (the marker is unknowable on the first attempt).
        results = [e["result"] for e in state["_rubric_evaluations"]]
        assert results[0] == "needs_revision", f"first review should fail, got {results}"
        assert results[-1] == "satisfied"
        assert len(results) >= 2, "loop did not re-draft"

        # 3. The final answer reflects both criteria.
        final = result["messages"][-1]
        assert isinstance(final, AIMessage)
        text = final.content if isinstance(final.content, str) else str(final.content)
        assert "Paris" in text
        assert "DONE-7Q2" in text

        # 4. The on_evaluation callback observed every review.
        assert [e["result"] for e in evals] == results


# ------------------------------------------------------------------
# User mode — the open loop must ask the user, then converge
# ------------------------------------------------------------------


class TestRubricUserModeConversation:
    async def test_agent_asks_user_then_satisfies(self):
        grader = RubricExtension(
            model=_llm(),
            mode="user",
            review=1,  # grade every turn
        )

        class UserAgent(Agent):
            model = _llm()
            extensions = [grader]
            prompt = "You are a helpful assistant collecting trip details."

            handler = staticmethod(_agentkit_handler)

        graph = await UserAgent().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "rubric-user"}}

        rubric = (
            "- The assistant has recorded the user's destination city\n"
            "- The assistant has recorded the user's travel month"
        )

        # Turn 1: the user gives only the city. The month criterion can only be
        # satisfied by asking — so the agent must yield with a question.
        out1 = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="I want to plan a trip to Lisbon.")],
                "rubric": rubric,
            },
            config=config,
        )
        state1 = (await graph.aget_state(config)).values

        assert state1["_rubric_status"] == "needs_revision", "turn 1 should not yet be satisfied"
        # The turn ended back to the user (no synthetic grader turn injected).
        assert isinstance(out1["messages"][-1], AIMessage)
        assert not out1["messages"][-1].tool_calls
        # The agent asked the user for the missing fact (best-effort signal).
        asked = out1["messages"][-1].content.lower()
        assert "?" in asked

        # Turn 2: the user supplies the month — now both criteria can be met.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Sometime in October.")]},
            config=config,
        )
        state2 = (await graph.aget_state(config)).values

        assert state2["_rubric_status"] == "satisfied", (
            f"expected satisfied after the user answered, got {state2['_rubric_status']}"
        )
        results = [e["result"] for e in state2["_rubric_evaluations"]]
        assert results[-1] == "satisfied"
        assert "needs_revision" in results[:-1]
