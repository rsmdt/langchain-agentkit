# ruff: noqa: N805
"""End-to-end tests for ``RubricExtension`` wired into a real graph.

Covers both modes. The grader is driven by a fake chat model whose
``with_structured_output`` channel returns canned ``GraderResponse`` verdicts.
The ``_rubric_*`` bookkeeping keys are ``PrivateStateAttr``, so they are read
from checkpointed state via ``get_state`` (see ``_run`` / ``_run_turns``).

A noop tool is registered so the graph wires the conditional edges that honor
``jump_to: "model"`` (the auto-mode revision loop routes through them).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_agentkit import Agent
from langchain_agentkit.extensions.rubric import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    GraderResponse,
    RubricExtension,
)


@tool
def noop(x: str) -> str:
    """Echo helper — gives the graph an executable tool so jump_to is honored."""
    return "ok"


def _main_model() -> MagicMock:
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


def _verdict(
    *, result: str, explanation: str, criteria: list[dict[str, Any]] | None = None
) -> GraderResponse:
    return GraderResponse(result=result, explanation=explanation, criteria=criteria or [])  # type: ignore[arg-type]


class _FakeGraderModel:
    """Fake grader driven through ``with_structured_output``; records prompts."""

    def __init__(self, *verdicts: Any) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.calls = 0
        self.system_prompts: list[str] = []

    def with_structured_output(self, schema: Any, **_: Any) -> _FakeGraderModel:  # noqa: ARG002
        return self

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        self.calls += 1
        self.system_prompts.extend(str(m.content) for m in messages if isinstance(m, SystemMessage))
        resp = self._verdicts[self._i]
        self._i += 1
        if isinstance(resp, BaseException):
            raise resp
        return resp


def _agent(handler_messages: list[Any], grader: Any, **rubric_kwargs: Any) -> type[Agent]:
    """Agent whose handler replays ``handler_messages`` (one per model call)."""
    calls = {"n": 0}

    class _RubricAgent(Agent):
        model = _main_model()
        tools = [noop]
        extensions = [RubricExtension(model=grader, **rubric_kwargs)]

        async def handler(state: Any, *, llm: Any) -> dict[str, Any]:  # noqa: ARG001
            msg = handler_messages[calls["n"]]
            calls["n"] += 1
            return {"messages": [msg], "sender": "rubric_agent"}

    return _RubricAgent


async def _run(agent_cls: type[Agent], invoke_input: dict[str, Any]) -> tuple[dict, dict]:
    """Single-invoke run; returns (output, checkpointed state)."""
    compiled = await agent_cls().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "rubric-flow"}}
    output = await compiled.ainvoke(invoke_input, config=config)
    state = (await compiled.aget_state(config)).values
    return output, state


async def _run_turns(agent_cls: type[Agent], turns: list[dict[str, Any]]) -> list[dict]:
    """Multi-invoke run on one thread; returns the checkpointed state after each turn."""
    compiled = await agent_cls().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "rubric-convo"}}
    states: list[dict] = []
    for turn in turns:
        await compiled.ainvoke(turn, config=config)
        states.append((await compiled.aget_state(config)).values)
    return states


# --------------------------------------------------------------------- #
# auto mode — closed loop
# --------------------------------------------------------------------- #


class TestAutoMode:
    async def test_satisfied_first_try(self):
        grader = _FakeGraderModel(_verdict(result="satisfied", explanation="ok"))
        agent_cls = _agent([AIMessage(content="draft")], grader, mode="auto")
        output, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- built"}
        )
        assert "_rubric_status" not in output  # private
        assert state["_rubric_status"] == "satisfied"
        assert state["_rubric_iterations"] == 1

    async def test_needs_revision_loops_back_then_satisfied(self):
        """The revision message is injected AND the model re-runs."""
        grader = _FakeGraderModel(
            _verdict(
                result="needs_revision",
                explanation="add tests",
                criteria=[{"name": "tests", "passed": False, "gap": "no tests"}],
            ),
            _verdict(result="satisfied", explanation="ok now"),
        )
        agent_cls = _agent(
            [AIMessage(content="first"), AIMessage(content="second")],
            grader,
            mode="auto",
            review={"stop": 5},
        )
        output, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- tests pass"}
        )
        injected = [
            m
            for m in output["messages"]
            if m.additional_kwargs.get("lc_source") == RUBRIC_GRADER_MESSAGE_SOURCE
        ]
        assert len(injected) == 1
        assert "no tests" in injected[0].content
        ai = [m.content for m in output["messages"] if isinstance(m, AIMessage)]
        assert "first" in ai and "second" in ai
        assert state["_rubric_status"] == "satisfied"
        assert [e["result"] for e in state["_rubric_evaluations"]] == [
            "needs_revision",
            "satisfied",
        ]

    async def test_review_limit_reached(self):
        grader = _FakeGraderModel(
            _verdict(
                result="needs_revision",
                explanation="x",
                criteria=[{"name": "a", "passed": False, "gap": "g"}],
            ),
            _verdict(
                result="needs_revision",
                explanation="x",
                criteria=[{"name": "a", "passed": False, "gap": "g"}],
            ),
        )
        agent_cls = _agent(
            [AIMessage(content="t1"), AIMessage(content="t2")],
            grader,
            mode="auto",
            review={"stop": 2},
        )
        _, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- thing"}
        )
        assert state["_rubric_status"] == "review_limit_reached"
        assert state["_rubric_iterations"] == 2

    async def test_no_rubric_is_noop(self):
        grader = _FakeGraderModel()
        agent_cls = _agent([AIMessage(content="hi")], grader, mode="auto")
        _, state = await _run(agent_cls, {"messages": [HumanMessage(content="hi")]})
        assert state.get("_rubric_status") is None
        assert state.get("_rubric_evaluations", []) == []

    async def test_custom_grader_system_prompt(self):
        grader = _FakeGraderModel(_verdict(result="satisfied", explanation="ok"))
        agent_cls = _agent(
            [AIMessage(content="draft")], grader, mode="auto", system_prompt="CUSTOM_MARKER strict"
        )
        await _run(agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- x"})
        assert any("CUSTOM_MARKER" in p for p in grader.system_prompts)


# --------------------------------------------------------------------- #
# user mode — open loop
# --------------------------------------------------------------------- #


class TestUserMode:
    async def test_grades_each_turn_and_yields_until_satisfied(self):
        grader = _FakeGraderModel(
            _verdict(
                result="needs_revision",
                explanation="more",
                criteria=[{"name": "a", "passed": False, "gap": "need detail"}],
            ),
            _verdict(result="satisfied", explanation="done"),
        )
        agent_cls = _agent(
            [AIMessage(content="draft 1"), AIMessage(content="draft 2")],
            grader,
            mode="user",
            review=1,  # grade every turn
        )
        states = await _run_turns(
            agent_cls,
            [
                {"messages": [HumanMessage(content="write X")], "rubric": "- detailed"},
                {"messages": [HumanMessage(content="here is detail")]},  # rubric sticks
            ],
        )
        # Turn 1: graded, not satisfied → yielded (status needs_revision, conversation continues).
        assert states[0]["_rubric_status"] == "needs_revision"
        # Turn 2: graded again → satisfied.
        assert states[1]["_rubric_status"] == "satisfied"
        assert [e["result"] for e in states[1]["_rubric_evaluations"]] == [
            "needs_revision",
            "satisfied",
        ]
        # No synthetic grader message was injected in user mode.
        assert grader.calls == 2

    async def test_window_defers_grading_until_max(self):
        # review max=3 → turns 1,2 are not graded; turn 3 is forced.
        grader = _FakeGraderModel(_verdict(result="satisfied", explanation="ok"))
        agent_cls = _agent(
            [AIMessage(content="t1"), AIMessage(content="t2"), AIMessage(content="t3")],
            grader,
            mode="user",
            review={"min": 1, "max": 3},
        )
        states = await _run_turns(
            agent_cls,
            [
                {"messages": [HumanMessage(content="start")], "rubric": "- crit"},
                {"messages": [HumanMessage(content="more")]},
                {"messages": [HumanMessage(content="even more")]},
            ],
        )
        # Turns 1 & 2: grader not called, just counting.
        assert grader.calls == 1
        assert states[0]["_rubric_turns_since_review"] == 1
        assert states[0].get("_rubric_evaluations", []) == []
        assert states[1]["_rubric_turns_since_review"] == 2
        # Turn 3: forced grade → satisfied, counter reset.
        assert states[2]["_rubric_status"] == "satisfied"
        assert states[2]["_rubric_turns_since_review"] == 0

    async def test_request_review_grades_early(self):
        # The agent calls request_review on its first model step; grading then
        # fires at the turn's natural stop instead of waiting for max=8.
        grader = _FakeGraderModel(_verdict(result="satisfied", explanation="ok"))
        agent_cls = _agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "request_review", "args": {}, "id": "r1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="final"),
            ],
            grader,
            mode="user",
            review={"min": 1, "max": 8},
        )
        _, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- crit"}
        )
        assert grader.calls == 1  # graded this turn despite max=8
        assert state["_rubric_status"] == "satisfied"
