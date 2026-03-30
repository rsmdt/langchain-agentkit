# ruff: noqa: N801
"""Real-LLM integration evals for TeamExtension coordination.

Tests exercise the FULL compiled graph flow: lead agent receives a message,
uses team coordination tools (AgentTeam, AssignTask, CheckTeammates,
DissolveTeam) via LangGraph's ReAct loop, teammates process with real LLM
calls as asyncio.Tasks, and results propagate back through the lead.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_team_extension_eval.py -x -v -m eval
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.tasks import TasksExtension
from langchain_agentkit.extensions.teams import TeamExtension

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
# Worker agent definitions
# ---------------------------------------------------------------------------


def _build_worker():
    """General-purpose worker that answers questions concisely."""
    _llm = _get_llm()

    class worker(agent):
        model = _llm
        description = "General-purpose worker that answers questions concisely"
        prompt = (
            "You are a helpful assistant. Answer questions concisely "
            "in one sentence. Be direct and factual."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "worker"}

    return worker


def _build_math_worker():
    """Math-focused worker that answers with just the number."""
    _llm = _get_llm()

    class math_worker(agent):
        model = _llm
        description = "Answers math questions with just the numeric result"
        prompt = (
            "You are a calculator. When asked a math question, respond with "
            "ONLY the numeric answer. No words, no explanation."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "math_worker"}

    return math_worker


# ---------------------------------------------------------------------------
# Lead agent builder
# ---------------------------------------------------------------------------

_TEAM_LEAD_PROMPT = """\
You are a team lead. Follow these steps EXACTLY in order:

1. Use AgentTeam to create a team with the workers you need.
   - Each member needs a unique "name" and an "agent_type" matching a registered agent.
2. Use AssignTask to give each worker their task.
3. Use CheckTeammates to collect results. If no pending messages yet, call \
CheckTeammates again after a moment.
4. Once you have all results, use DissolveTeam to shut down the team.
5. Report the final results to the user clearly.

IMPORTANT: Always complete ALL steps. Never skip DissolveTeam.
IMPORTANT: You MUST call the tools — never answer questions yourself.\
"""


def _build_team_lead(mw_team, mw_tasks):
    """Build a lead agent with team and task extensions."""
    _llm = _get_llm()

    class team_lead(agent):
        model = _llm
        extensions = [mw_tasks, mw_team]
        prompt = _TEAM_LEAD_PROMPT

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "team_lead"}

    return team_lead


# ---------------------------------------------------------------------------
# Test 1: Single worker full lifecycle via compiled graph
# ---------------------------------------------------------------------------


class TestTeamSingleWorkerLifecycle:
    """Lead spawns 1 worker, assigns task, checks, dissolves — full graph flow."""

    async def test_team_single_worker_lifecycle(self):
        """Lead creates team, assigns 'Capital of Japan?', reports 'Tokyo'."""
        worker = _build_worker()
        mw_team = TeamExtension([worker])
        mw_tasks = TasksExtension()
        lead = _build_team_lead(mw_team, mw_tasks)

        graph = lead.compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'geo-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Assign alice the task: 'What is the capital of Japan?' "
                            "Then check for her answer and dissolve the team. "
                            "Report alice's answer."
                        )
                    )
                ]
            },
            {"recursion_limit": 40},
        )

        # Team should be dissolved (team_name reset to None)
        assert result.get("team_name") is None, (
            f"Expected team dissolved (team_name=None), got: {result.get('team_name')}"
        )

        # Final response should contain "Tokyo"
        final = result["messages"][-1].content.lower()
        assert "tokyo" in final, f"Expected 'tokyo' in final response, got: {final}"


# ---------------------------------------------------------------------------
# Test 2: Multi-agent team via compiled graph
# ---------------------------------------------------------------------------


class TestTeamMultiAgent:
    """Lead spawns 2 workers of different types, assigns parallel tasks."""

    async def test_team_multi_agent(self):
        """Worker answers factual question, math_worker answers math."""
        worker = _build_worker()
        math_worker = _build_math_worker()
        mw_team = TeamExtension([worker, math_worker])
        mw_tasks = TasksExtension()
        lead = _build_team_lead(mw_team, mw_tasks)

        graph = lead.compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'mixed-team' with two members:\n"
                            "- 'alice' of agent_type 'worker'\n"
                            "- 'bob' of agent_type 'math_worker'\n"
                            "Assign alice: 'What color is the sky on a clear day?'\n"
                            "Assign bob: 'What is 15 + 27?'\n"
                            "Check for their answers, dissolve the team, and "
                            "report both results."
                        )
                    )
                ]
            },
            {"recursion_limit": 50},
        )

        # Team should be dissolved
        assert result.get("team_name") is None, (
            f"Expected team dissolved, got: {result.get('team_name')}"
        )

        # Final response should contain both answers
        final = result["messages"][-1].content.lower()
        assert "blue" in final, f"Expected 'blue' in final response, got: {final}"
        assert "42" in final, f"Expected '42' in final response, got: {final}"


# ---------------------------------------------------------------------------
# Test 3: Tools exposed (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamToolsExposed:
    """Verify TeamExtension exposes exactly 5 tools."""

    def test_team_tools_exposed(self):
        worker = _build_worker()
        mw = TeamExtension([worker])
        tool_names = sorted(t.name for t in mw.tools)

        assert tool_names == [
            "AgentTeam",
            "AssignTask",
            "CheckTeammates",
            "DissolveTeam",
            "MessageTeammate",
        ]


# ---------------------------------------------------------------------------
# Test 4: State schema (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamStateSchema:
    """Verify state_schema returns TeamState."""

    def test_team_state_schema(self):
        from langchain_agentkit.state import TeamState

        worker = _build_worker()
        mw = TeamExtension([worker])
        assert mw.state_schema is TeamState


# ---------------------------------------------------------------------------
# Test 5: Prompt includes agent roster (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamPromptRoster:
    """Verify prompt() includes registered agent names."""

    def test_team_prompt_roster(self):
        worker = _build_worker()
        mw = TeamExtension([worker])
        prompt = mw.prompt(state={"messages": []})

        assert "worker" in prompt
