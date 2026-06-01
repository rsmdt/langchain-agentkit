"""HITL on a fully-assembled AgentKit — deterministic, no LLM.

Proves the HITL redesign composes with a realistic extension stack
(Resilience + History + Filesystem + HITL), both as a standalone agent and
as a subgraph of a supervisor. The interesting interaction is
``ResilienceExtension``: it sits outermost in the tool-hook chain and catches
tool exceptions into synthetic ``ToolMessage``s, but re-raises ``GraphBubbleUp``
so an ``interrupt()`` still pauses the graph (``resilience/extension.py:137``).

Covered:

* tool approval through the full stack — standalone and as a subgraph
* AskUser — its in-ToolNode ``interrupt()`` survives Resilience's wrap
* Resilience self-heal (a raising tool → synthetic error message) coexisting
  with an approval interrupt in the same run
"""

from __future__ import annotations

import tempfile
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from langchain_agentkit import (
    Agent,
    CountStrategy,
    FilesystemExtension,
    HistoryExtension,
    HITLExtension,
    ResilienceExtension,
)
from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.extensions.hitl import InterruptConfig

pytestmark = pytest.mark.asyncio


class StubModel(BaseChatModel):
    """Deterministic model that replays a preset list of AIMessages."""

    responses: list[AIMessage] = []
    idx: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> StubModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        msg = self.responses[min(self.idx, len(self.responses) - 1)]
        type(self).idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "stub"


@tool
def record(note: str) -> str:
    """Record a note (gated tool used for approval tests)."""
    return f"RECORDED {note}"


@tool
def boom(x: str) -> str:
    """A tool that always raises — exercises Resilience self-heal."""
    raise RuntimeError(f"kaboom: {x}")


def _full_stack_agent(
    responses: list[AIMessage], *, gate_record: bool = True, with_history: bool = True
) -> Agent:
    """Assemble Resilience + (History) + Filesystem + HITL on one agent.

    Resilience is first so it is the outermost tool-hook layer.

    ``with_history`` defaults on for standalone agents. Subgraph callers pass
    ``with_history=False``: ``HistoryExtension`` rewrites the shared ``messages``
    channel in ``wrap_model``, which breaks ``interrupt()`` resume when the kit
    runs as a subgraph — a pre-existing limitation independent of HITL, captured
    by ``test_history_as_subgraph_breaks_interrupt_resume`` below.
    """
    StubModel.idx = 0
    tmpdir = tempfile.mkdtemp(prefix="hitl_fullstack_")

    extensions: list[Any] = [ResilienceExtension()]
    if with_history:
        extensions.append(HistoryExtension(strategy=CountStrategy(max_messages=100)))
    extensions.append(FilesystemExtension(backend=OSBackend(root=tmpdir)))
    extensions.append(
        HITLExtension(
            interrupt_on=(
                {"record": InterruptConfig(options=["approve", "reject"])} if gate_record else {}
            ),
        )
    )

    agent = Agent()
    agent.model = StubModel(responses=responses)
    agent.tools = [record, boom]
    agent.extensions = extensions
    agent.prompt = "test agent"

    async def handler(
        state: dict[str, Any], *, llm: Any, tools: Any, prompt: str, runtime: Any
    ) -> dict[str, Any]:
        bound = llm.bind_tools(tools)
        msgs = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await bound.ainvoke(msgs)]}

    agent.handler = handler  # type: ignore[method-assign]
    return agent


def _ai(*tool_calls: dict[str, Any], content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


def _record_call(cid: str = "rec_1", note: str = "hello") -> dict[str, Any]:
    return {"name": "record", "args": {"note": note}, "id": cid, "type": "tool_call"}


def _boom_call(cid: str = "boom_1") -> dict[str, Any]:
    return {"name": "boom", "args": {"x": "now"}, "id": cid, "type": "tool_call"}


def _ask_call(cid: str = "ask_1") -> dict[str, Any]:
    return {
        "name": "AskUser",
        "args": {
            "questions": [
                {
                    "question": "Which path?",
                    "header": "Path",
                    "options": [
                        {"label": "Fast", "description": "f"},
                        {"label": "Safe", "description": "s"},
                    ],
                    "multi_select": False,
                }
            ]
        },
        "id": cid,
        "type": "tool_call",
    }


def _tool_messages(result: dict[str, Any]) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}


class _SupState(TypedDict):
    messages: Annotated[list[Any], add_messages]


def _supervisor(subgraph: Any) -> Any:
    sup = StateGraph(_SupState)
    sup.add_node("worker", subgraph)
    sup.add_edge(START, "worker")
    sup.add_edge("worker", END)
    return sup.compile(checkpointer=InMemorySaver())


# ------------------------------------------------------------------
# (a) Fully-assembled standalone AgentKit node
# ------------------------------------------------------------------


async def test_full_stack_approval_standalone():
    agent = _full_stack_agent([_ai(_record_call()), _ai(content="done")])
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "fs-approve"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record hello")]}, cfg)
    state = await graph.aget_state(cfg)
    assert len(state.interrupts) == 1
    assert state.interrupts[0].value["questions"][0]["context"]["tool"] == "record"

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)
    assert "RECORDED hello" in _tool_messages(result)["rec_1"].content


async def test_full_stack_reject_standalone():
    agent = _full_stack_agent([_ai(_record_call()), _ai(content="done")])
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "fs-reject"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record hello")]}, cfg)
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Reject"}}), cfg)

    tm = _tool_messages(result)["rec_1"]
    assert tm.status == "error"
    assert "RECORDED" not in tm.content


async def test_full_stack_askuser_survives_resilience():
    """AskUser's interrupt is raised inside the ToolNode, directly wrapped by
    Resilience. It must pause the graph, not become a synthetic error message."""
    agent = _full_stack_agent([_ai(_ask_call()), _ai(content="done")], gate_record=False)
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "fs-ask"}}

    await graph.ainvoke({"messages": [HumanMessage(content="ask me")]}, cfg)
    state = await graph.aget_state(cfg)
    # Paused on a real interrupt — not swallowed into an error ToolMessage.
    assert len(state.interrupts) == 1
    assert state.interrupts[0].value["questions"][0]["question"] == "Which path?"

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Fast"}}), cfg)
    assert "Which path? → Fast" in _tool_messages(result)["ask_1"].content


async def test_full_stack_resilience_selfheal_coexists_with_interrupt():
    """A gated call (interrupt) and a raising call (Resilience self-heal) in one
    run: the interrupt pauses, and on resume the raising tool yields a synthetic
    error message instead of crashing the graph."""
    agent = _full_stack_agent([_ai(_record_call(), _boom_call()), _ai(content="done")])
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "fs-coexist"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record and boom")]}, cfg)
    state = await graph.aget_state(cfg)
    # Only the gated call is in the interrupt; boom is not gated.
    assert len(state.interrupts[0].value["questions"]) == 1

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)
    tms = _tool_messages(result)
    assert "RECORDED hello" in tms["rec_1"].content  # approved tool executed
    boom_msg = tms["boom_1"]
    assert boom_msg.status == "error"  # raising tool -> Resilience synthetic message
    assert boom_msg.additional_kwargs.get("agentkit", {}).get("synthesized") is True


# ------------------------------------------------------------------
# (b) Fully-assembled AgentKit as a subgraph of a supervisor
# ------------------------------------------------------------------


async def test_full_stack_approval_as_subgraph():
    # History excluded — see _full_stack_agent / the xfail test below.
    agent = _full_stack_agent([_ai(_record_call()), _ai(content="done")], with_history=False)
    subgraph = await agent.compile()  # no checkpointer — inherits supervisor's
    graph = _supervisor(subgraph)
    cfg = {"configurable": {"thread_id": "fs-sub-approve"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record hello")]}, cfg)
    state = graph.get_state(cfg)
    assert len(state.interrupts) == 1
    assert state.interrupts[0].value["questions"][0]["context"]["tool"] == "record"

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)
    assert any(
        isinstance(m, ToolMessage) and "RECORDED hello" in m.content for m in result["messages"]
    )


async def test_full_stack_askuser_as_subgraph_survives_resilience():
    agent = _full_stack_agent(
        [_ai(_ask_call()), _ai(content="done")], gate_record=False, with_history=False
    )
    subgraph = await agent.compile()
    graph = _supervisor(subgraph)
    cfg = {"configurable": {"thread_id": "fs-sub-ask"}}

    await graph.ainvoke({"messages": [HumanMessage(content="ask me")]}, cfg)
    state = graph.get_state(cfg)
    assert len(state.interrupts) == 1

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Safe"}}), cfg)
    assert any(
        isinstance(m, ToolMessage) and "Which path? → Safe" in m.content for m in result["messages"]
    )


async def test_full_stack_subgraph_coexist_resilience_and_interrupt():
    agent = _full_stack_agent(
        [_ai(_record_call(), _boom_call()), _ai(content="done")], with_history=False
    )
    subgraph = await agent.compile()
    graph = _supervisor(subgraph)
    cfg = {"configurable": {"thread_id": "fs-sub-coexist"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record and boom")]}, cfg)
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)

    tms = _tool_messages(result)
    assert "RECORDED hello" in tms["rec_1"].content
    assert tms["boom_1"].status == "error"
    assert tms["boom_1"].additional_kwargs.get("agentkit", {}).get("synthesized") is True


# ------------------------------------------------------------------
# Unsupported composition — HistoryExtension inside a subgraph
# ------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Unsupported by design: HistoryExtension replaces the shared 'messages' "
        "channel via ReplaceMessages (a clear-all RemoveMessage). Inside a subgraph "
        "that clear-all escapes to the parent and wipes its messages, so the "
        "supervisor re-schedules the worker and the agent re-interrupts instead of "
        "resuming. Independent of HITL — reproduces with a bare interrupt() tool. "
        "Recommended composition: keep History out of the embedded kit and truncate "
        "in the outer graph (docs/subgraph-composition.md). AgentKit logs a one-time "
        "warning when this is detected."
    ),
)
async def test_history_as_subgraph_breaks_interrupt_resume():
    """Pins the unsupported composition; strict-xfail flips if the behavior changes."""
    agent = _full_stack_agent([_ai(_record_call()), _ai(content="done")], with_history=True)
    subgraph = await agent.compile()
    graph = _supervisor(subgraph)
    cfg = {"configurable": {"thread_id": "fs-sub-history"}}

    await graph.ainvoke({"messages": [HumanMessage(content="record hello")]}, cfg)
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)

    # Expected to FAIL today: resume does not execute the tool through History.
    assert any(
        isinstance(m, ToolMessage) and "RECORDED hello" in m.content for m in result["messages"]
    )
