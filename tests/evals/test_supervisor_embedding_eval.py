"""LLM-behaviour eval: an embedded AgentKit agent decides when to hand back.

Companion to ``tests/integration/test_supervisor_embedding.py``, which proves
the *wiring* deterministically. This proves the *behaviour*: a real LLM agent
embedded as a node in a supervisor graph correctly calls ``complete_phase`` vs
``return_to_supervisor`` based on whether the request belongs to its phase.

Because an active hand-off (``Command(graph=PARENT)``) keeps the agent's
tool-call ``AIMessage`` inside the subgraph, the stock trajectory extractor
would see no tool calls at the parent. Behaviour is therefore observed via which
parent node control reached: ``advance`` (completed) vs ``classify_route``
(handed back) — captured in the supervisor's ``visited`` trail.

Run with::

    pytest tests/evals/test_supervisor_embedding_eval.py -v -m eval
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_agentkit import Agent
from tests._supervisor_embedding import (
    SETUP_PROMPT,
    build_supervisor,
    do_setup_work,
    handoff_tools,
    orphaned_tool_messages,
)
from tests.evals.conftest import EVAL_MODEL

# Mark all tests in this module as eval (skipped unless explicitly run).
pytestmark = pytest.mark.eval

try:
    from langchain_openai import ChatOpenAI

    _HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _HAS_OPENAI = False

skip_reason = "Requires OPENAI_API_KEY and langchain-openai"


async def _build_llm_supervisor() -> Any:
    """A supervisor embedding a real LLM-driven setup agent as a node."""
    llm = ChatOpenAI(model=EVAL_MODEL, temperature=0)

    class SetupAgent(Agent):
        model = llm
        prompt = SETUP_PROMPT
        tools = [do_setup_work, *handoff_tools()]
        max_turns = 6

        async def handler(state, *, llm, tools, prompt):  # noqa: N805
            from langchain_core.messages import SystemMessage

            bound = llm.bind_tools(tools)
            messages = [SystemMessage(content=prompt), *state["messages"]]
            return {"messages": [await bound.ainvoke(messages)]}

    agent = await SetupAgent().compile()
    return build_supervisor(agent)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestSupervisorEmbeddingBehaviour:
    """A real LLM agent embedded in a supervisor hands off to the right place."""

    async def test_on_phase_request_completes_and_advances(self) -> None:
        """A setup-phase request -> the agent works and routes the parent to 'advance'."""
        supervisor = await _build_llm_supervisor()

        result = await supervisor.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Create the initial project scaffolding: a src/ "
                        "directory and a pyproject.toml."
                    )
                ]
            }
        )

        assert "advance" in result["visited"], (
            f"expected completion -> advance, visited={result['visited']}"
        )
        assert "classify_route" not in result["visited"]
        # The hand-off fired (its closing ToolMessage is in the parent history)...
        assert any(
            isinstance(m, ToolMessage) and "complete_phase->advance" in m.content
            for m in result["messages"]
        )
        # ...and the worker's forwarded history is complete and well-formed:
        # the agent's setup work is preserved, and no ToolMessage is orphaned, so
        # the supervisor's next LLM call over these messages would be valid.
        assert any(isinstance(m, ToolMessage) for m in result["messages"])
        assert orphaned_tool_messages(result["messages"]) == []

    async def test_off_phase_request_hands_back_to_supervisor(self) -> None:
        """A non-setup request -> the agent routes the parent to 'classify_route'."""
        supervisor = await _build_llm_supervisor()

        result = await supervisor.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Summarize the latest quarterly sales figures for me.")
                ]
            }
        )

        assert "classify_route" in result["visited"], (
            f"expected hand-back -> classify_route, visited={result['visited']}"
        )
        assert "advance" not in result["visited"]
        # The hand-back fired and forwarded a complete, well-formed history.
        assert any(
            isinstance(m, ToolMessage) and "return_to_supervisor->classify_route" in m.content
            for m in result["messages"]
        )
        assert orphaned_tool_messages(result["messages"]) == []
