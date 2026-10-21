from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.agents.types import AgentConfig, _AgentConfigProxy

_ID_KEY = "agentkit_subagent_id"


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **_: Any) -> _ScriptedModel:
        return self

    async def ainvoke(self, messages: list[BaseMessage], *_: Any, **__: Any) -> AIMessage:
        self.calls.append(list(messages))
        return self._responses.pop(0)


@tool
def approve(action: str) -> str:
    """Request approval for an action."""
    answer = interrupt({"action": action})
    return f"approved={answer}"


def _parent(
    configs: list[AgentConfig],
    models: dict[str, _ScriptedModel],
    *,
    checkpointer: Any = None,
    ephemeral: bool = False,
) -> Any:
    tools: list[Any] = [approve]
    default_model = next(iter(models.values()))
    tools.extend(
        create_agent_tools(
            agents_by_name={c.name: _AgentConfigProxy(c) for c in configs},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=lambda: tools,
            ephemeral=ephemeral,
            parent_llm_getter=lambda: default_model,
            model_resolver=lambda name: models[name],
        )
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools[1:]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile(checkpointer=checkpointer)


def _call(call_id: str, **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "Agent", "args": args, "id": call_id}])


def _thread(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _result_for(messages: list[BaseMessage], call_id: str) -> ToolMessage:
    return next(m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id)


def _agent_id(result: ToolMessage) -> str:
    agent_id = result.response_metadata[_ID_KEY]
    assert agent_id in str(result.content)
    return str(agent_id)


@pytest.mark.asyncio
async def test_child_interrupt_surfaces_at_parent_and_resume_completes_child():
    model = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "approve", "args": {"action": "deploy"}, "id": "a1"}],
            ),
            AIMessage(content="child done"),
        ]
    )
    config = AgentConfig(name="helper", description="d", prompt="p", tools=["approve"])
    app = _parent([config], {"helper": model}, checkpointer=InMemorySaver())
    thread = _thread("t1")

    first = await app.ainvoke(
        {"messages": [_call("c1", agent={"id": "helper"}, message="go")]}, thread
    )

    assert [i.value for i in first["__interrupt__"]] == [{"action": "deploy"}]
    snapshot = await app.aget_state(thread)
    assert [i.value for t in snapshot.tasks for i in t.interrupts] == [{"action": "deploy"}]
    assert not any(isinstance(m, ToolMessage) for m in first["messages"])

    resumed = await app.ainvoke(Command(resume="yes"), thread)

    assert "__interrupt__" not in resumed
    result = _result_for(resumed["messages"], "c1")
    assert result.status == "success"
    assert "child done" in str(result.content)
    approval = model.calls[-1][-1]
    assert isinstance(approval, ToolMessage)
    assert approval.content == "approved=yes"


@pytest.mark.asyncio
async def test_completed_sibling_is_not_rerun_when_parent_resumes():
    quick = _ScriptedModel([AIMessage(content="quick done")])
    gated = _ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "approve", "args": {"action": "x"}, "id": "a1"}],
            ),
            AIMessage(content="gated done"),
        ]
    )
    configs = [
        AgentConfig(name="quick", description="d", prompt="p", model="quick"),
        AgentConfig(name="gated", description="d", prompt="p", model="gated", tools=["approve"]),
    ]
    app = _parent(configs, {"quick": quick, "gated": gated}, checkpointer=InMemorySaver())
    thread = _thread("t1")
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "Agent", "args": {"agent": {"id": "quick"}, "message": "a"}, "id": "c1"},
            {"name": "Agent", "args": {"agent": {"id": "gated"}, "message": "b"}, "id": "c2"},
        ],
    )

    first = await app.ainvoke({"messages": [both]}, thread)
    assert first["__interrupt__"]
    resumed = await app.ainvoke(Command(resume="ok"), thread)

    assert len(quick.calls) == 1
    assert "quick done" in str(_result_for(resumed["messages"], "c1").content)
    assert "gated done" in str(_result_for(resumed["messages"], "c2").content)


@pytest.mark.asyncio
async def test_resume_continues_isolated_child_with_full_history():
    model = _ScriptedModel([AIMessage(content="first answer"), AIMessage(content="second answer")])
    config = AgentConfig(name="helper", description="d", prompt="p")
    app = _parent([config], {"helper": model}, checkpointer=InMemorySaver())
    thread = _thread("t1")

    first = await app.ainvoke(
        {"messages": [_call("c1", agent={"id": "helper"}, message="first task")]}, thread
    )
    agent_id = _agent_id(_result_for(first["messages"], "c1"))

    second = await app.ainvoke(
        {"messages": [_call("c2", agent={"id": "helper"}, message="follow up", resume=agent_id)]},
        thread,
    )

    result = _result_for(second["messages"], "c2")
    assert result.status == "success"
    assert "second answer" in str(result.content)
    assert _agent_id(result) == agent_id
    seen = model.calls[1]
    assert [type(m) for m in seen] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert "first task" in str(seen[1].content)
    assert seen[2].content == "first answer"
    assert "follow up" in str(seen[3].content)


@pytest.mark.asyncio
async def test_resume_continues_forked_child_without_resending_parent_history():
    model = _ScriptedModel([AIMessage(content="first answer"), AIMessage(content="second answer")])
    config = AgentConfig(name="helper", description="d", prompt="p", mode="fork")
    app = _parent([config], {"helper": model}, checkpointer=InMemorySaver())
    thread = _thread("t1")
    history = [HumanMessage(content="The account id is 4471."), AIMessage(content="Noted.")]

    first = await app.ainvoke(
        {"messages": [*history, _call("c1", agent={"id": "helper"}, message="summarize")]},
        thread,
    )
    agent_id = _agent_id(_result_for(first["messages"], "c1"))
    await app.ainvoke(
        {"messages": [_call("c2", agent={"id": "helper"}, message="and more", resume=agent_id)]},
        thread,
    )

    first_view, second_view = model.calls
    assert second_view[: len(first_view)] == first_view
    assert [type(m) for m in second_view[len(first_view) :]] == [AIMessage, HumanMessage]
    assert "and more" in str(second_view[-1].content)


@pytest.mark.asyncio
async def test_dynamic_child_is_resumable():
    model = _ScriptedModel([AIMessage(content="one"), AIMessage(content="two")])
    app = _parent([], {"dyn": model}, checkpointer=InMemorySaver(), ephemeral=True)
    thread = _thread("t1")
    agent = {"prompt": "You are terse."}

    first = await app.ainvoke({"messages": [_call("c1", agent=agent, message="hi")]}, thread)
    agent_id = _agent_id(_result_for(first["messages"], "c1"))
    await app.ainvoke(
        {"messages": [_call("c2", agent=agent, message="again", resume=agent_id)]}, thread
    )

    assert [type(m) for m in model.calls[1]] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thread_id", "resume", "agent"),
    [
        ("other-thread", None, "helper"),
        ("t1", "not-an-id", "helper"),
        ("t1", None, "other"),
    ],
)
async def test_resume_rejects_unknown_foreign_or_mismatched_ids(
    thread_id: str, resume: str | None, agent: str
):
    model = _ScriptedModel([AIMessage(content="first answer")])
    configs = [
        AgentConfig(name="helper", description="d", prompt="p"),
        AgentConfig(name="other", description="d", prompt="p"),
    ]
    app = _parent(configs, {"helper": model}, checkpointer=InMemorySaver())

    first = await app.ainvoke(
        {"messages": [_call("c1", agent={"id": "helper"}, message="task")]}, _thread("t1")
    )
    agent_id = resume or _agent_id(_result_for(first["messages"], "c1"))

    second = await app.ainvoke(
        {"messages": [_call("c2", agent={"id": agent}, message="more", resume=agent_id)]},
        _thread(thread_id),
    )

    assert _result_for(second["messages"], "c2").status == "error"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_without_checkpointer_results_carry_no_id_and_resume_is_an_error():
    model = _ScriptedModel([AIMessage(content="answer")])
    config = AgentConfig(name="helper", description="d", prompt="p")
    app = _parent([config], {"helper": model})

    first = await app.ainvoke({"messages": [_call("c1", agent={"id": "helper"}, message="t")]})
    second = await app.ainvoke(
        {"messages": [_call("c2", agent={"id": "helper"}, message="t", resume="abc")]}
    )

    assert _ID_KEY not in _result_for(first["messages"], "c1").response_metadata
    assert _result_for(second["messages"], "c2").status == "error"
    assert len(model.calls) == 1
