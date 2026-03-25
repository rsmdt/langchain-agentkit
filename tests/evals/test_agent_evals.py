"""Integration evals — run an actual LLM agent and verify tool usage.

These tests require:
- An LLM API key (OPENAI_API_KEY or similar)
- The ``eval`` optional dependency group

Run::

    pytest tests/evals/test_agent_evals.py -v -m eval --tb=short

Skip with::

    pytest tests/evals/ -v -m "not eval"
"""

import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.graph.message import add_messages

from langchain_agentkit.state import _merge_tasks
from tests.evals.datasets import (
    MULTI_STEP_DATASET,
    READ_TOOL_DATASET,
    SKILL_LOADING_DATASET,
    TASK_CREATE_DATASET,
    TASK_DEPENDENCIES_DATASET,
    TASK_LIFECYCLE_DATASET,
    TASK_LIST_DATASET,
    TASK_STOP_DATASET,
)
from tests.evals.eval_runner import (
    print_eval_results,
    run_eval,
)

# Mark all tests in this module as eval (skipped unless explicitly run)
pytestmark = pytest.mark.eval

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Skip entire module if no API key or langchain-openai unavailable
try:
    from langchain_openai import ChatOpenAI

    _HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
except ImportError:
    _HAS_OPENAI = False

skip_reason = "Requires OPENAI_API_KEY and langchain-openai"


# State schemas at module level for Python 3.14 get_type_hints() compatibility
class _SkillsEvalState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]


class _TasksEvalState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    tasks: Annotated[list[dict[str, Any]], _merge_tasks]


def _build_skills_agent():
    """Build a ReAct agent with skills + filesystem tools."""
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from langchain_agentkit import AgentKit, SkillsMiddleware

    mw = SkillsMiddleware(str(FIXTURES / "skills"))
    kit = AgentKit([mw])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    bound_llm = llm.bind_tools(kit.tools)

    def agent_node(state: _SkillsEvalState) -> dict:
        system = SystemMessage(content=kit.prompt(state))
        messages = [system] + state["messages"]
        return {"messages": [bound_llm.invoke(messages)]}

    def should_continue(state: _SkillsEvalState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(_SkillsEvalState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(kit.tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


def _build_tasks_agent():
    """Build a ReAct agent with task management tools."""
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from langchain_agentkit import AgentKit, TasksMiddleware

    mw = TasksMiddleware()
    kit = AgentKit([mw])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    bound_llm = llm.bind_tools(kit.tools)

    def agent_node(state: _TasksEvalState) -> dict:
        system = SystemMessage(content=kit.prompt(state))
        messages = [system] + state["messages"]
        return {"messages": [bound_llm.invoke(messages)]}

    def should_continue(state: _TasksEvalState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(_TasksEvalState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(kit.tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Skills + Filesystem Evals
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestSkillLoadingEval:
    """Verify agent loads skills for domain-specific requests."""

    def test_skill_loading_dataset(self):
        agent = _build_skills_agent()
        results = run_eval(
            agent=agent,
            dataset=SKILL_LOADING_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestReadToolEval:
    """Verify agent uses Read tool for file access."""

    def test_read_tool_dataset(self):
        agent = _build_skills_agent()
        results = run_eval(
            agent=agent,
            dataset=READ_TOOL_DATASET,
            trajectory_mode="subset",
            tool_args_mode="subset",
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestMultiStepEval:
    """Verify agent chains multiple tool calls correctly."""

    def test_multi_step_dataset(self):
        agent = _build_skills_agent()
        results = run_eval(
            agent=agent,
            dataset=MULTI_STEP_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        passed = sum(1 for r in results if r["score"])
        total = len(results)
        assert passed >= total * 0.5, (
            f"Only {passed}/{total} multi-step evals passed"
        )


# ---------------------------------------------------------------------------
# Task Tool Evals
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestTaskCreateEval:
    """Verify agent creates tasks for multi-step requests."""

    def test_creates_tasks_for_list(self):
        agent = _build_tasks_agent()
        results = run_eval(
            agent=agent,
            dataset=TASK_CREATE_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestTaskLifecycleEval:
    """Verify agent follows task lifecycle: create → in_progress → completed."""

    def test_task_lifecycle(self):
        agent = _build_tasks_agent()
        results = run_eval(
            agent=agent,
            dataset=TASK_LIFECYCLE_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestTaskDependenciesEval:
    """Verify agent creates tasks with dependency ordering."""

    def test_task_dependencies(self):
        agent = _build_tasks_agent()
        results = run_eval(
            agent=agent,
            dataset=TASK_DEPENDENCIES_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        # Dependencies are hard — allow some flexibility
        passed = sum(1 for r in results if r["score"])
        total = len(results)
        assert passed >= total * 0.5, (
            f"Only {passed}/{total} dependency evals passed"
        )


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestTaskListEval:
    """Verify agent lists tasks when asked for status."""

    def test_task_list(self):
        agent = _build_tasks_agent()
        results = run_eval(
            agent=agent,
            dataset=TASK_LIST_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


@pytest.mark.skipif(not _HAS_OPENAI, reason=skip_reason)
class TestTaskStopEval:
    """Verify agent stops a running task."""

    def test_task_stop(self):
        agent = _build_tasks_agent()

        # Pre-seed state with a running task so TaskStop has something to stop
        results = run_eval(
            agent=agent,
            dataset=TASK_STOP_DATASET,
            trajectory_mode="subset",
            tool_args_mode="ignore",
            state_factory=lambda: {
                "messages": [
                    {"role": "user", "content": TASK_STOP_DATASET[0]["inputs"]},
                ],
                "tasks": [
                    {
                        "id": "running-1",
                        "subject": "Long analysis",
                        "description": "Running analysis",
                        "status": "in_progress",
                        "active_form": "Analyzing...",
                    },
                ],
            },
        )
        print_eval_results(results)
        for r in results:
            assert r["score"], _fail_msg(r)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fail_msg(r: dict) -> str:
    return (
        f"FAILED: {r['description']}\n"
        f"  Comment: {r['comment']}\n"
        f"  Expected: {r['expected_tool_calls']}\n"
        f"  Actual: {r['actual_tool_calls']}"
    )
