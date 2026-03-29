"""Unified Agent tool for LangGraph agent delegation.

Tools return ``Command(update={"messages": [ToolMessage(...)]})`` to
propagate delegation results. Compatible with LangGraph's ``ToolNode``
out of the box.

Usage::

    from langchain_agentkit.tools.agent import create_agent_tools

    tools = create_agent_tools(
        agents_by_name={"researcher": researcher_graph},
        compiled_cache={},
        delegation_timeout=300.0,
        parent_tools_getter=lambda: parent_tools,
        ephemeral=True,
        parent_llm_getter=lambda: llm,
    )
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.graph import END
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Agent reference types (discriminated by field shape)
# ---------------------------------------------------------------------------


class Predefined(BaseModel):
    """Select a pre-defined agent from the roster."""

    id: str = Field(description="Agent name from the available roster.")


class Dynamic(BaseModel):
    """Create an on-the-fly reasoning agent."""

    prompt: str = Field(
        description="System prompt defining the agent's role and behavior.",
    )


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _AgentInputBase(BaseModel):
    name: str | None = Field(
        default=None,
        description="Human-readable label for this delegation (shown in UI).",
    )
    message: str = Field(
        description=(
            "Clear, specific task for the agent. Include all necessary "
            "context — the agent has no access to your conversation history."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _AgentInput(_AgentInputBase):
    """Input schema when only pre-defined agents are available."""

    agent: Predefined = Field(
        description="The pre-defined agent to delegate to.",
    )


class _AgentDynamicInput(_AgentInputBase):
    """Input schema when both pre-defined and dynamic agents are available."""

    agent: Predefined | Dynamic = Field(
        description=(
            "The agent to delegate to. Use {id} for a pre-defined agent "
            "from the roster, or {prompt} to create a custom reasoning agent."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scoped_state(message: str, sender: str = "parent") -> dict[str, Any]:
    """Build an isolated state with only the delegation message."""
    return {
        "messages": [HumanMessage(content=message)],
        "sender": sender,
    }


def _extract_final_response(result: dict[str, Any]) -> str:
    """Extract the final AI message content from an invocation result."""
    messages = result.get("messages", [])
    if not messages:
        return "(no response)"
    last = messages[-1]
    content = getattr(last, "content", str(last))
    return content if content else "(empty response)"


def _compile_agent(
    graph: Any,
    compiled_cache: dict[str, Any],
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
) -> Any:
    """Compile an agent graph, caching the result.

    Handles ``tools="inherit"`` by injecting parent tools before compilation.
    """
    name: str = graph.agentkit_name
    if name in compiled_cache:
        return compiled_cache[name]

    if getattr(graph, "agentkit_tools_inherit", False) and parent_tools_getter is not None:
        parent_tools = parent_tools_getter()
        if parent_tools:
            tool_node = ToolNode(parent_tools)
            if "tools" in graph.nodes:
                graph.nodes["tools"] = tool_node
            else:
                graph.add_node("tools", tool_node)
                agent_node_name = name
                graph.add_conditional_edges(
                    agent_node_name,
                    lambda state: "tools" if (
                        state["messages"][-1].tool_calls
                        if hasattr(state["messages"][-1], "tool_calls")
                        else False
                    ) else END,
                    {"tools": "tools", END: END},
                )
                graph.add_edge("tools", agent_node_name)

    compiled = graph.compile()
    compiled_cache[name] = compiled
    return compiled


# ---------------------------------------------------------------------------
# Shared delegation runner
# ---------------------------------------------------------------------------


async def _run_delegation(
    compiled: Any,
    scoped_state: dict[str, Any],
    agent: str,
    timeout: float,
    tool_call_id: str,
) -> Any:
    """Invoke a compiled graph with timeout and error handling."""
    try:
        result = await asyncio.wait_for(compiled.ainvoke(scoped_state), timeout=timeout)
    except TimeoutError:
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"Delegation to '{agent}' timed out after {timeout}s",
                    tool_call_id=tool_call_id,
                ),
            ],
        })
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        return Command(update={
            "messages": [
                ToolMessage(
                    content=f"Delegation failed: {error_msg}",
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    response = _extract_final_response(result)
    return Command(update={
        "messages": [
            ToolMessage(content=response, tool_call_id=tool_call_id),
        ],
    })


# ---------------------------------------------------------------------------
# Unified tool implementation
# ---------------------------------------------------------------------------


async def _agent_tool(
    agent: dict[str, Any] | Predefined | Dynamic,
    message: str,
    state: dict[str, Any],
    tool_call_id: str,
    name: str | None = None,
    *,
    agents_by_name: dict[str, Any],
    compiled_cache: dict[str, Any],
    delegation_timeout: float,
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
    ephemeral: bool,
    parent_llm_getter: Callable[[], Any] | None,
) -> Any:
    """Delegate a task to a pre-defined or dynamically created agent."""
    # Normalise to dict — LangChain may pass a Pydantic model or a raw dict
    agent_ref = agent if isinstance(agent, dict) else agent.model_dump()

    if "id" in agent_ref:
        return await _delegate_predefined(
            agent_id=agent_ref["id"],
            message=message,
            agents_by_name=agents_by_name,
            compiled_cache=compiled_cache,
            delegation_timeout=delegation_timeout,
            parent_tools_getter=parent_tools_getter,
            tool_call_id=tool_call_id,
        )

    if "prompt" in agent_ref:
        if not ephemeral:
            raise ToolException(
                "Dynamic agents are not enabled. "
                "Set ephemeral=True on AgentExtension to allow custom agents."
            )
        return await _delegate_dynamic(
            prompt=agent_ref["prompt"],
            message=message,
            delegation_timeout=delegation_timeout,
            parent_llm_getter=parent_llm_getter,
            tool_call_id=tool_call_id,
        )

    raise ToolException(
        "Invalid agent reference. Provide {id} for a pre-defined agent "
        "or {prompt} for a custom agent."
    )


async def _delegate_predefined(
    agent_id: str,
    message: str,
    agents_by_name: dict[str, Any],
    compiled_cache: dict[str, Any],
    delegation_timeout: float,
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
    tool_call_id: str,
) -> Any:
    """Delegate a task to a named pre-defined agent."""
    from langchain_agentkit.extensions import resolve_agent

    graph = resolve_agent(agent_id, agents_by_name)
    compiled = _compile_agent(graph, compiled_cache, parent_tools_getter)
    scoped_state = _build_scoped_state(message)

    return await _run_delegation(
        compiled=compiled,
        scoped_state=scoped_state,
        agent=agent_id,
        timeout=delegation_timeout,
        tool_call_id=tool_call_id,
    )


async def _delegate_dynamic(
    prompt: str,
    message: str,
    delegation_timeout: float,
    parent_llm_getter: Callable[[], Any] | None,
    tool_call_id: str,
) -> Any:
    """Delegate a task to a dynamically created reasoning agent."""
    if not prompt or not prompt.strip():
        raise ToolException("Agent prompt cannot be empty.")

    if parent_llm_getter is None:
        raise ToolException("Dynamic agents require a parent LLM.")

    llm = parent_llm_getter()
    ephemeral_name = "dynamic"

    from langchain_agentkit._graph_builder import build_graph
    from langchain_agentkit.agent_kit import AgentKit

    async def _ephemeral_handler(
        state_inner: dict[str, Any],
        *,
        llm: Any,
        prompt: str,
    ) -> dict[str, Any]:
        from langchain_core.messages import SystemMessage

        msgs = [SystemMessage(content=prompt)] + list(
            state_inner.get("messages", [])
        )
        response = await llm.ainvoke(msgs)
        return {"messages": [response], "sender": ephemeral_name}

    try:
        kit = AgentKit([], prompt=prompt)
        graph = build_graph(
            name=ephemeral_name,
            handler=_ephemeral_handler,
            llm=llm,
            user_tools=[],
            kit=kit,
            state_type=AgentKitState,
        )
        compiled = graph.compile()
    except Exception as exc:
        raise ToolException(
            f"Failed to create dynamic agent: {exc}"
        ) from exc

    scoped_state = _build_scoped_state(message)

    return await _run_delegation(
        compiled=compiled,
        scoped_state=scoped_state,
        agent=ephemeral_name,
        timeout=delegation_timeout,
        tool_call_id=tool_call_id,
    )


# ---------------------------------------------------------------------------
# Tool description
# ---------------------------------------------------------------------------


_AGENT_DESCRIPTION = """\
Delegate a task to an agent and wait for the result.

Use when:
- The task requires specialized tools you don't have
- The task is independent and can be done in isolation
- You need multiple things done in parallel (call Agent multiple times in one turn)
- The task would benefit from focused, context-isolated execution

Do NOT use when:
- The task requires your full conversation context
- The task is trivial (faster to do yourself)
- The task requires back-and-forth discussion

Provide a clear, self-contained message. The agent receives ONLY your message — \
it has no access to your conversation history.\
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_agent_tools(
    agents_by_name: dict[str, Any],
    compiled_cache: dict[str, Any],
    delegation_timeout: float,
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
    ephemeral: bool,
    parent_llm_getter: Callable[[], Any] | None,
) -> list[BaseTool]:
    """Create the unified Agent tool for agent-to-agent delegation.

    Returns a list containing the Agent tool. When ``ephemeral=True``,
    the tool schema includes the Dynamic agent variant.

    Args:
        agents_by_name: Dict mapping agent name to its StateGraph.
        compiled_cache: Shared cache for compiled graphs (mutated in place).
        delegation_timeout: Max seconds to wait for a subagent response.
        parent_tools_getter: Callable returning parent's tools (for inherit).
        ephemeral: Whether to include the Dynamic agent variant in the schema.
        parent_llm_getter: Callable returning the parent LLM (for dynamic).
    """
    if ephemeral and parent_llm_getter is None:
        msg = "parent_llm_getter is required when ephemeral=True"
        raise ValueError(msg)

    schema = _AgentDynamicInput if ephemeral else _AgentInput

    bound = partial(
        _agent_tool,
        agents_by_name=agents_by_name,
        compiled_cache=compiled_cache,
        delegation_timeout=delegation_timeout,
        parent_tools_getter=parent_tools_getter,
        ephemeral=ephemeral,
        parent_llm_getter=parent_llm_getter,
    )

    tool = StructuredTool.from_function(
        coroutine=bound,
        name="Agent",
        description=_AGENT_DESCRIPTION,
        args_schema=schema,
        handle_tool_error=True,
    )

    return [tool]
