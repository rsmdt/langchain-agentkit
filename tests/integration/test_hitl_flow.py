"""Deterministic HITL interrupt/resume proof — no LLM, fully reproducible.

These tests pin down the human-in-the-loop contract end-to-end against a
scripted stub model, so the interrupt is triggered on every run without an
API key. They are the reproducible proof that backs the LLM evals in
``tests/evals/test_hitl_eval.py``.

Covered:

* approve / edit / reject / respond decisions on a gated tool
* parallel gated calls — one batched interrupt, decisions routed by index
  (the case that collides under a per-tool interrupt)
* AskUser — multi-question interrupt and index-keyed readback
* AgentKit graph embedded as a subgraph: the interrupt bubbles up to the
  supervisor and ``Command(resume=...)`` routes back down into the subgraph
"""

from __future__ import annotations

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

from langchain_agentkit import Agent, HITLExtension
from langchain_agentkit.extensions.hitl import InterruptConfig

pytestmark = pytest.mark.asyncio


# ------------------------------------------------------------------
# Scripted stub model — emits a fixed queue of AIMessages, ignores tools
# ------------------------------------------------------------------


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
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    return f"WROTE {path}={content!r}"


def _ai(*tool_calls: dict[str, Any], content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


def _tc(cid: str, path: str = "/a", content: str = "x") -> dict[str, Any]:
    return {
        "name": "write_file",
        "args": {"path": path, "content": content},
        "id": cid,
        "type": "tool_call",
    }


def _build_agent(
    responses: list[AIMessage], options: list[str], *, ask_user: bool = False
) -> Agent:
    StubModel.idx = 0
    agent = Agent()
    agent.model = StubModel(responses=responses)
    agent.tools = [write_file]
    agent.extensions = [
        HITLExtension(
            interrupt_on={"write_file": InterruptConfig(options=options)},  # type: ignore[arg-type]
            tools=None if ask_user else [],
        )
    ]
    agent.prompt = "test"

    async def handler(
        state: dict[str, Any], *, llm: Any, tools: Any, prompt: str, runtime: Any
    ) -> dict[str, Any]:
        bound = llm.bind_tools(tools)
        return {
            "messages": [await bound.ainvoke([SystemMessage(content=prompt)] + state["messages"])]
        }

    agent.handler = handler  # type: ignore[method-assign]
    return agent


async def _run(graph: Any, cfg: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    await graph.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)
    return await graph.ainvoke(Command(resume={"answers": answers}), cfg)


def _tool_messages(result: dict[str, Any]) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}


# ------------------------------------------------------------------
# Single gated call — each decision type
# ------------------------------------------------------------------


async def test_approve_executes_tool():
    agent = _build_agent([_ai(_tc("c1")), _ai(content="done")], ["approve", "reject"])
    graph = await agent.compile(checkpointer=InMemorySaver())
    result = await _run(graph, {"configurable": {"thread_id": "approve"}}, {"0": "Approve"})

    tm = _tool_messages(result)["c1"]
    assert tm.status != "error"
    assert "WROTE /a" in tm.content


async def test_reject_blocks_tool():
    agent = _build_agent([_ai(_tc("c1")), _ai(content="done")], ["approve", "reject"])
    graph = await agent.compile(checkpointer=InMemorySaver())
    result = await _run(graph, {"configurable": {"thread_id": "reject"}}, {"0": "Reject"})

    tm = _tool_messages(result)["c1"]
    assert tm.status == "error"
    assert not any("WROTE" in m.content for m in _tool_messages(result).values())


async def test_edit_executes_with_revised_args():
    agent = _build_agent(
        [_ai(_tc("c1")), _ai(content="done")], ["approve", "edit", "reject", "respond"]
    )
    graph = await agent.compile(checkpointer=InMemorySaver())
    result = await _run(
        graph,
        {"configurable": {"thread_id": "edit"}},
        {"0": {"type": "edit", "args": {"path": "/EDITED", "content": "z"}}},
    )

    tm = _tool_messages(result)["c1"]
    assert "WROTE /EDITED" in tm.content
    assert "'z'" in tm.content


async def test_respond_answers_on_behalf_without_executing():
    agent = _build_agent([_ai(_tc("c1")), _ai(content="done")], ["approve", "reject", "respond"])
    graph = await agent.compile(checkpointer=InMemorySaver())
    result = await _run(
        graph,
        {"configurable": {"thread_id": "respond"}},
        {"0": {"type": "respond", "message": "used cached result"}},
    )

    tm = _tool_messages(result)["c1"]
    assert tm.status == "success"
    assert tm.content == "used cached result"
    assert not any("WROTE" in m.content for m in _tool_messages(result).values())


# ------------------------------------------------------------------
# Parallel gated calls — one batched interrupt, decisions routed by index
# ------------------------------------------------------------------


async def test_parallel_calls_single_interrupt_correct_routing():
    parallel = _ai(_tc("call_a", path="/a"), _tc("call_b", path="/b"))
    agent = _build_agent([parallel, _ai(content="done")], ["approve", "reject"])
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "parallel"}}

    await graph.ainvoke({"messages": [HumanMessage(content="write both")]}, cfg)

    # Exactly one interrupt, carrying both questions.
    state = await graph.aget_state(cfg)
    assert len(state.interrupts) == 1
    assert len(state.interrupts[0].value["questions"]) == 2

    # Approve call_a, reject call_b — each decision lands on the right call.
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve", "1": "Reject"}}), cfg)
    tms = _tool_messages(result)
    assert tms["call_a"].status != "error" and "WROTE /a" in tms["call_a"].content
    assert tms["call_b"].status == "error"


async def test_mixed_gated_and_ungated_calls():
    @tool
    def search(q: str) -> str:
        """Search."""
        return f"FOUND {q}"

    StubModel.idx = 0
    agent = Agent()
    agent.model = StubModel(
        responses=[
            _ai(
                _tc("gated", path="/x"),
                {"name": "search", "args": {"q": "cats"}, "id": "free", "type": "tool_call"},
            ),
            _ai(content="done"),
        ]
    )
    agent.tools = [write_file, search]
    agent.extensions = [HITLExtension(interrupt_on={"write_file": True}, tools=[])]
    agent.prompt = "test"

    async def handler(state, *, llm, tools, prompt, runtime):
        bound = llm.bind_tools(tools)
        return {
            "messages": [await bound.ainvoke([SystemMessage(content=prompt)] + state["messages"])]
        }

    agent.handler = handler  # type: ignore[method-assign]
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "mixed"}}

    await graph.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)
    # Only the gated call is in the interrupt.
    state = await graph.aget_state(cfg)
    assert len(state.interrupts[0].value["questions"]) == 1

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Reject"}}), cfg)
    tms = _tool_messages(result)
    assert tms["gated"].status == "error"  # gated -> rejected
    assert "FOUND cats" in tms["free"].content  # ungated -> executed normally


# ------------------------------------------------------------------
# AskUser — multi-question interrupt and index-keyed readback
# ------------------------------------------------------------------


async def test_ask_user_multi_question_interrupt_and_resume():
    ask_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "AskUser",
                "args": {
                    "questions": [
                        {
                            "question": "Which DB?",
                            "header": "DB",
                            "options": [
                                {"label": "Postgres", "description": "rel"},
                                {"label": "Mongo", "description": "doc"},
                            ],
                            "multi_select": False,
                        },
                        {
                            "question": "Which region?",
                            "header": "Region",
                            "options": [
                                {"label": "EU", "description": "eu"},
                                {"label": "US", "description": "us"},
                            ],
                            "multi_select": False,
                        },
                    ]
                },
                "id": "ask_1",
                "type": "tool_call",
            }
        ],
    )
    StubModel.idx = 0
    agent = Agent()
    agent.model = StubModel(responses=[ask_call, AIMessage(content="done")])
    agent.tools = []
    agent.extensions = [HITLExtension()]  # AskUser, no gating
    agent.prompt = "test"

    async def handler(state, *, llm, tools, prompt, runtime):
        bound = llm.bind_tools(tools)
        return {
            "messages": [await bound.ainvoke([SystemMessage(content=prompt)] + state["messages"])]
        }

    agent.handler = handler  # type: ignore[method-assign]
    graph = await agent.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "ask"}}

    await graph.ainvoke({"messages": [HumanMessage(content="help me choose")]}, cfg)
    state = await graph.aget_state(cfg)
    assert len(state.interrupts) == 1
    assert len(state.interrupts[0].value["questions"]) == 2

    result = await graph.ainvoke(Command(resume={"answers": {"0": "Postgres", "1": "US"}}), cfg)
    readback = _tool_messages(result)["ask_1"].content
    assert "Which DB? → Postgres" in readback
    assert "Which region? → US" in readback


# ------------------------------------------------------------------
# AgentKit as a subgraph of a supervisor — interrupt bubbles up & resumes down
# ------------------------------------------------------------------


class _SupState(TypedDict):
    messages: Annotated[list[Any], add_messages]


async def test_agentkit_subgraph_interrupt_bubbles_and_resumes():
    agent = _build_agent([_ai(_tc("sub_call")), _ai(content="subdone")], ["approve", "reject"])
    # Compile WITHOUT a checkpointer — the subgraph inherits the supervisor's.
    subgraph = await agent.compile()

    supervisor = StateGraph(_SupState)
    supervisor.add_node("agentkit", subgraph)
    supervisor.add_edge(START, "agentkit")
    supervisor.add_edge("agentkit", END)
    graph = supervisor.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "subgraph"}}

    await graph.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)

    # The subgraph's interrupt is visible on the supervisor's state.
    state = graph.get_state(cfg)
    assert len(state.interrupts) == 1
    assert state.interrupts[0].value["questions"][0]["context"]["tool"] == "write_file"

    # Resuming on the supervisor routes the decision down into the subgraph.
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), cfg)
    assert any(isinstance(m, ToolMessage) and "WROTE" in m.content for m in result["messages"])


async def test_agentkit_subgraph_reject_blocks_in_supervisor():
    agent = _build_agent([_ai(_tc("sub_call")), _ai(content="subdone")], ["approve", "reject"])
    subgraph = await agent.compile()

    supervisor = StateGraph(_SupState)
    supervisor.add_node("agentkit", subgraph)
    supervisor.add_edge(START, "agentkit")
    supervisor.add_edge("agentkit", END)
    graph = supervisor.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "subgraph-reject"}}

    await graph.ainvoke({"messages": [HumanMessage(content="go")]}, cfg)
    result = await graph.ainvoke(Command(resume={"answers": {"0": "Reject"}}), cfg)

    tms = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tms and all(m.status == "error" for m in tms)
    assert not any("WROTE" in m.content for m in tms)
