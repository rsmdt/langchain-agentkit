from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.agents.types import AgentConfig, _AgentConfigProxy


class _ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools: Any, **_: Any) -> _ScriptedModel:
        return self

    async def ainvoke(self, messages: list[BaseMessage], *_: Any, **__: Any) -> AIMessage:
        self.calls.append(list(messages))
        return self._responses.pop(0)


def _agent_tools(config: AgentConfig, model: _ScriptedModel) -> list[Any]:
    tools: list[Any] = []
    tools.extend(
        create_agent_tools(
            agents_by_name={config.name: _AgentConfigProxy(config)},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=lambda: tools,
            ephemeral=False,
            parent_llm_getter=lambda: model,
        )
    )
    return tools


async def _delegate(tools: list[Any], history: list[BaseMessage], message: str) -> list[Any]:
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "Agent", "args": {"agent": {"id": "helper"}, "message": message}, "id": "c1"}
        ],
    )
    result = await graph.compile().ainvoke({"messages": [*history, call]})
    return list(result["messages"])


_HISTORY: list[BaseMessage] = [
    HumanMessage(content="The account id is 4471."),
    AIMessage(content="Noted."),
    AIMessage(content="old subagent step", response_metadata={"agentkit_hidden_from_llm": True}),
]


@pytest.mark.asyncio
async def test_isolated_agent_sees_only_the_delegated_message():
    model = _ScriptedModel([AIMessage(content="done")])
    config = AgentConfig(name="helper", description="d", prompt="sub prompt")

    await _delegate(_agent_tools(config, model), _HISTORY, "summarize the account")

    seen = model.calls[0]
    assert [type(m) for m in seen] == [SystemMessage, HumanMessage]
    assert seen[1].content == "summarize the account"


@pytest.mark.asyncio
async def test_fork_inherits_visible_history_and_receives_framed_instruction():
    model = _ScriptedModel([AIMessage(content="done")])
    config = AgentConfig(name="helper", description="d", prompt="sub prompt", mode="fork")

    messages = await _delegate(_agent_tools(config, model), _HISTORY, "summarize the account")

    seen = model.calls[0]
    assert [type(m) for m in seen] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert seen[1].content == "The account id is 4471."
    assert seen[2].content == "Noted."
    assert not any(getattr(m, "tool_calls", None) for m in seen)
    assert seen[3].content.endswith("summarize the account")
    assert seen[3].content != "summarize the account"
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "success"
    assert tool_msg.content == "done"


@pytest.mark.asyncio
async def test_fork_cannot_delegate_again():
    nested_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Agent",
                "args": {"agent": {"id": "helper"}, "message": "go deeper"},
                "id": "nested",
            }
        ],
    )
    model = _ScriptedModel([nested_call, AIMessage(content="did it myself")])
    config = AgentConfig(
        name="helper", description="d", prompt="sub prompt", tools=["Agent"], mode="fork"
    )

    messages = await _delegate(_agent_tools(config, model), _HISTORY, "summarize the account")

    assert len(model.calls) == 2
    refusal = model.calls[1][-1]
    assert isinstance(refusal, ToolMessage)
    assert refusal.tool_call_id == "nested"
    assert refusal.status == "error"
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.content == "did it myself"
