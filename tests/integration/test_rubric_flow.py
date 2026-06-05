# ruff: noqa: N805
"""End-to-end tests for ``RubricExtension`` wired into a real graph.

The main agent's handler returns canned ``AIMessage``s; the grader is driven
by a fake chat model whose ``with_structured_output`` channel returns canned
``GraderResponse`` verdicts (the provider-agnostic channel the grader uses in
production). The ``_rubric_*`` bookkeeping keys are ``PrivateStateAttr``, so
they are absent from the ``ainvoke`` output and are read from the checkpointed
state via ``get_state`` (see ``_run``).

A noop tool is registered on each agent so the graph wires the conditional
edges that honor ``jump_to: "model"`` — the revision loop routes through them.
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
    """A mock main model; the handler returns canned messages and never calls it."""
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    return mock


def _verdict(
    *,
    result: str,
    explanation: str,
    criteria: list[dict[str, Any]] | None = None,
) -> GraderResponse:
    """A grader verdict, delivered via the structured-output channel."""
    return GraderResponse(result=result, explanation=explanation, criteria=criteria or [])  # type: ignore[arg-type]


class _FakeGraderModel:
    """Fake grader model driven through ``with_structured_output``.

    These flows configure no grader tools, so only the verdict channel is
    exercised. ``ainvoke`` replays the canned verdicts in order and records
    the system prompts it was handed.
    """

    def __init__(self, *verdicts: Any) -> None:
        self._verdicts = list(verdicts)
        self._i = 0
        self.system_prompts: list[str] = []

    def with_structured_output(self, schema: Any, **_: Any) -> _FakeGraderModel:  # noqa: ARG002
        return self

    async def ainvoke(self, messages: Any, **_: Any) -> Any:
        self.system_prompts.extend(str(m.content) for m in messages if isinstance(m, SystemMessage))
        resp = self._verdicts[self._i]
        self._i += 1
        if isinstance(resp, BaseException):
            raise resp
        return resp


def _agent(handler_messages: list[AIMessage], grader: Any, **rubric_kwargs: Any) -> type[Agent]:
    """Build an Agent subclass whose handler replays ``handler_messages`` in order."""
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
    """Compile with a checkpointer, invoke, and return (output, full state).

    The ``_rubric_*`` bookkeeping keys are ``PrivateStateAttr``, so they are
    absent from the ``ainvoke`` output and must be read from the checkpointed
    state via ``get_state``.
    """
    compiled = await agent_cls().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "rubric-flow"}}
    output = await compiled.ainvoke(invoke_input, config=config)
    state = (await compiled.aget_state(config)).values
    return output, state


class TestRubricFlow:
    async def test_satisfied_first_try_terminates(self):
        grader = _FakeGraderModel(
            _verdict(
                result="satisfied",
                explanation="looks good",
                criteria=[{"name": "built", "passed": True}],
            )
        )
        agent_cls = _agent([AIMessage(content="here is my draft")], grader, max_iterations=3)
        output, state = await _run(
            agent_cls,
            {"messages": [HumanMessage(content="do it")], "rubric": "- The thing is built"},
        )

        # Private bookkeeping is absent from the public output.
        assert "_rubric_status" not in output
        assert "_rubric_evaluations" not in output
        # No synthetic revision turn was injected.
        assert not any(
            m.additional_kwargs.get("lc_source") == RUBRIC_GRADER_MESSAGE_SOURCE
            for m in output["messages"]
        )
        # Bookkeeping is observable via get_state.
        assert state["_rubric_status"] == "satisfied"
        assert state["_rubric_iterations"] == 1
        assert len(state["_rubric_evaluations"]) == 1
        assert state["_rubric_evaluations"][0]["result"] == "satisfied"

    async def test_needs_revision_loops_back_then_satisfied(self):
        """The gap-gated test: needs_revision injects feedback AND re-runs the model."""
        grader = _FakeGraderModel(
            _verdict(
                result="needs_revision",
                explanation="add tests",
                criteria=[{"name": "tests", "passed": False, "gap": "no tests"}],
            ),
            _verdict(result="satisfied", explanation="ok now"),
        )
        agent_cls = _agent(
            [AIMessage(content="first attempt"), AIMessage(content="second attempt with fix")],
            grader,
            max_iterations=5,
        )
        output, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- tests pass"}
        )

        # The grader-injected revision message survived the jump and carries the tag.
        injected = [
            m
            for m in output["messages"]
            if m.additional_kwargs.get("lc_source") == RUBRIC_GRADER_MESSAGE_SOURCE
        ]
        assert len(injected) == 1
        assert injected[0].name == RUBRIC_GRADER_MESSAGE_SOURCE
        assert "add tests" in injected[0].content
        assert "no tests" in injected[0].content

        # Both main-model attempts reached the transcript.
        ai_contents = [m.content for m in output["messages"] if isinstance(m, AIMessage)]
        assert "first attempt" in ai_contents
        assert "second attempt with fix" in ai_contents

        assert state["_rubric_status"] == "satisfied"
        assert state["_rubric_iterations"] == 2
        assert [e["result"] for e in state["_rubric_evaluations"]] == [
            "needs_revision",
            "satisfied",
        ]

    async def test_max_iterations_reached(self):
        grader = _FakeGraderModel(
            _verdict(
                result="needs_revision",
                explanation="still missing",
                criteria=[{"name": "x", "passed": False, "gap": "y"}],
            ),
            _verdict(
                result="needs_revision",
                explanation="still missing",
                criteria=[{"name": "x", "passed": False, "gap": "y"}],
            ),
        )
        agent_cls = _agent(
            [AIMessage(content="attempt 1"), AIMessage(content="attempt 2")],
            grader,
            max_iterations=2,
        )
        _, state = await _run(
            agent_cls, {"messages": [HumanMessage(content="do it")], "rubric": "- thing"}
        )

        assert state["_rubric_status"] == "max_iterations_reached"
        assert state["_rubric_iterations"] == 2
        assert len(state["_rubric_evaluations"]) == 2
        assert all(e["result"] == "needs_revision" for e in state["_rubric_evaluations"])

    async def test_no_rubric_is_noop(self):
        # An empty grader iterator raises IndexError if ever invoked.
        grader = _FakeGraderModel()
        agent_cls = _agent([AIMessage(content="hello")], grader, max_iterations=3)
        output, state = await _run(agent_cls, {"messages": [HumanMessage(content="say hi")]})

        assert not any(
            m.additional_kwargs.get("lc_source") == RUBRIC_GRADER_MESSAGE_SOURCE
            for m in output["messages"]
        )
        assert state.get("_rubric_status") is None
        assert state.get("_rubric_iterations", 0) == 0
        assert state.get("_rubric_evaluations", []) == []

    # NOTE: the "KeyboardInterrupt must propagate, not become grader_error"
    # contract is covered at the unit level
    # (test_rubric.py::test_keyboard_interrupt_propagates), which asserts the
    # extension's ``except Exception`` does not swallow ``BaseException``. We
    # don't repeat it end-to-end because a KeyboardInterrupt raised inside
    # LangGraph's per-node ``asyncio.create_task`` aborts the pytest session
    # rather than round-tripping through ``pytest.raises`` — an asyncio
    # artifact unrelated to this extension.

    async def test_custom_grader_system_prompt_is_honored(self):
        grader = _FakeGraderModel(_verdict(result="satisfied", explanation="ok"))
        custom = "CUSTOM_GRADER_MARKER: be extremely strict about every criterion."
        agent_cls = _agent(
            [AIMessage(content="draft")], grader, system_prompt=custom, max_iterations=3
        )
        compiled = await agent_cls().compile()
        await compiled.ainvoke(
            {"messages": [HumanMessage(content="do it")], "rubric": "- whatever"}
        )

        assert grader.system_prompts, "expected the grader to receive a system prompt"
        assert "CUSTOM_GRADER_MARKER" in grader.system_prompts[0]
