"""Command-based agent delegation tools for LangGraph agents.

Tools use ``InjectedState`` to access graph state and return
``Command(update={"delegation_log": [...]})`` to record delegation
results. Compatible with LangGraph's ``ToolNode`` out of the box.

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
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel, Field

from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _DelegateInput(BaseModel):
    agent: str = Field(description="Name of the agent to delegate to.")
    message: str = Field(
        description=(
            "Clear, specific task for the subagent. Include all necessary "
            "context — the agent has no access to your conversation history."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _DelegateEphemeralInput(BaseModel):
    message: str = Field(
        description="Task for the ephemeral agent to perform.",
    )
    instructions: str = Field(
        description=(
            "System prompt defining the ephemeral agent's role and behavior. "
            "Must be non-empty."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_scoped_state(message: str) -> dict[str, Any]:
    """Build an isolated state with only the delegation message."""
    return {
        "messages": [HumanMessage(content=message)],
        "sender": "lead",
    }


def _extract_final_response(result: dict[str, Any]) -> str:
    """Extract the final AI message content from an invocation result."""
    messages = result.get("messages", [])
    if not messages:
        return "(no response)"
    last = messages[-1]
    content = getattr(last, "content", str(last))
    return content if content else "(empty response)"


def _build_log_entry(
    agent_name: str,
    message: str,
    result_summary: str,
    duration: float,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a delegation log entry."""
    entry: dict[str, Any] = {
        "agent_name": agent_name,
        "message": message,
        "result_summary": result_summary[:200],
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "duration_seconds": round(duration, 3),
    }
    if error is not None:
        entry["error"] = error
    return entry


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
            # Rebuild the graph's ToolNode with parent tools
            # The graph is a StateGraph — add tools before compiling
            tool_node = ToolNode(parent_tools)
            # Check if there's already a tools node to replace
            if "tools" in graph.nodes:
                graph.nodes["tools"] = tool_node
            else:
                # Add tool node and wire edges for the ReAct loop
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
# Tool implementations
# ---------------------------------------------------------------------------


async def _delegate(
    agent: str,
    message: str,
    state: dict[str, Any],
    tool_call_id: str,
    *,
    agents_by_name: dict[str, Any],
    compiled_cache: dict[str, Any],
    delegation_timeout: float,
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
) -> Any:
    """Delegate a task to a named subagent."""
    if agent not in agents_by_name:
        available = ", ".join(sorted(agents_by_name.keys()))
        raise ToolException(
            f"Agent '{agent}' not found. Available agents: {available}"
        )

    graph = agents_by_name[agent]
    compiled = _compile_agent(graph, compiled_cache, parent_tools_getter)
    scoped_state = _build_scoped_state(message)

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            compiled.ainvoke(scoped_state),
            timeout=delegation_timeout,
        )
    except asyncio.TimeoutError:
        duration = time.monotonic() - start
        log_entry = _build_log_entry(
            agent, message, "(timeout)", duration, error="timeout"
        )
        raise ToolException(
            f"Agent '{agent}' timed out after {delegation_timeout}s"
        ) from None
    except Exception as exc:
        duration = time.monotonic() - start
        error_msg = f"{type(exc).__name__}: {exc}"
        log_entry = _build_log_entry(
            agent, message, error_msg[:200], duration, error=error_msg
        )
        from langgraph.types import Command

        return Command(
            update={
                "delegation_log": [log_entry],
                "messages": [
                    ToolMessage(
                        content=f"Delegation to '{agent}' failed: {error_msg}",
                        tool_call_id=tool_call_id,
                    ),
                ],
            }
        )

    duration = time.monotonic() - start
    response = _extract_final_response(result)
    log_entry = _build_log_entry(agent, message, response, duration)

    from langgraph.types import Command

    return Command(
        update={
            "delegation_log": [log_entry],
            "messages": [
                ToolMessage(content=response, tool_call_id=tool_call_id),
            ],
        }
    )


async def _delegate_ephemeral(
    message: str,
    instructions: str,
    state: dict[str, Any],
    tool_call_id: str,
    *,
    delegation_timeout: float,
    parent_llm_getter: Callable[[], Any],
) -> Any:
    """Delegate a task to a temporary reasoning-only agent."""
    if not instructions or not instructions.strip():
        raise ToolException("instructions cannot be empty")

    llm = parent_llm_getter()
    ephemeral_name = "ephemeral"

    # Build a minimal reasoning-only ReAct graph
    async def _ephemeral_node(
        state_inner: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        from langchain_core.messages import SystemMessage

        msgs = [SystemMessage(content=instructions)] + list(
            state_inner.get("messages", [])
        )
        response = await llm.ainvoke(msgs)
        return {"messages": [response], "sender": ephemeral_name}

    try:
        workflow: StateGraph[Any] = StateGraph(AgentKitState)
        workflow.add_node(ephemeral_name, _ephemeral_node)
        workflow.set_entry_point(ephemeral_name)
        workflow.add_edge(ephemeral_name, END)
        compiled = workflow.compile()
    except Exception as exc:
        raise ToolException(
            f"Failed to create ephemeral agent: {exc}"
        ) from exc

    scoped_state = _build_scoped_state(message)

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            compiled.ainvoke(scoped_state),
            timeout=delegation_timeout,
        )
    except asyncio.TimeoutError:
        duration = time.monotonic() - start
        raise ToolException(
            f"Ephemeral agent timed out after {delegation_timeout}s"
        ) from None
    except Exception as exc:
        duration = time.monotonic() - start
        error_msg = f"{type(exc).__name__}: {exc}"
        log_entry = _build_log_entry(
            ephemeral_name, message, error_msg[:200], duration, error=error_msg
        )
        from langgraph.types import Command

        return Command(
            update={
                "delegation_log": [log_entry],
                "messages": [
                    ToolMessage(
                        content=f"Ephemeral delegation failed: {error_msg}",
                        tool_call_id=tool_call_id,
                    ),
                ],
            }
        )

    duration = time.monotonic() - start
    response = _extract_final_response(result)
    log_entry = _build_log_entry(ephemeral_name, message, response, duration)

    from langgraph.types import Command

    return Command(
        update={
            "delegation_log": [log_entry],
            "messages": [
                ToolMessage(content=response, tool_call_id=tool_call_id),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------


_DELEGATE_DESCRIPTION = """\
Delegate a task to a specialist agent and wait for the result.

Use when:
- The task requires specialized tools you don't have
- The task is independent and can be done in isolation
- You need multiple things done in parallel (call Delegate multiple times in one turn)
- The task would benefit from focused, context-isolated execution

Do NOT use when:
- The task requires your full conversation context
- The task is trivial (faster to do yourself)
- The task requires back-and-forth discussion

Provide a clear, self-contained message. The agent receives ONLY your message — \
it has no access to your conversation history.\
"""

_DELEGATE_EPHEMERAL_DESCRIPTION = """\
Create a temporary reasoning-only agent with custom instructions and delegate a task to it.

Use when:
- You need analysis or reasoning with a specific perspective or role
- No existing specialist agent fits the need
- The task is purely analytical (no tools needed)

The ephemeral agent is reasoning-only — it cannot use tools. It receives your \
instructions as its system prompt and your message as the task.\
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
    """Create delegation tools for agent-to-agent communication.

    Returns a list containing the Delegate tool and optionally
    the DelegateEphemeral tool (if ``ephemeral=True``).

    Args:
        agents_by_name: Dict mapping agent name to its StateGraph.
        compiled_cache: Shared cache for compiled graphs (mutated in place).
        delegation_timeout: Max seconds to wait for a subagent response.
        parent_tools_getter: Callable returning parent's tools (for inherit).
        ephemeral: Whether to include the DelegateEphemeral tool.
        parent_llm_getter: Callable returning the parent LLM (for ephemeral).
    """

    async def _delegate_fn(
        agent: str,
        message: str,
        state: dict[str, Any],
        tool_call_id: str,
    ) -> Any:
        return await _delegate(
            agent=agent,
            message=message,
            state=state,
            tool_call_id=tool_call_id,
            agents_by_name=agents_by_name,
            compiled_cache=compiled_cache,
            delegation_timeout=delegation_timeout,
            parent_tools_getter=parent_tools_getter,
        )

    delegate_tool = StructuredTool.from_function(
        coroutine=_delegate_fn,
        name="Delegate",
        description=_DELEGATE_DESCRIPTION,
        args_schema=_DelegateInput,
        handle_tool_error=True,
    )

    tools: list[BaseTool] = [delegate_tool]

    if ephemeral:
        if parent_llm_getter is None:
            msg = "parent_llm_getter is required when ephemeral=True"
            raise ValueError(msg)

        async def _delegate_ephemeral_fn(
            message: str,
            instructions: str,
            state: dict[str, Any],
            tool_call_id: str,
        ) -> Any:
            return await _delegate_ephemeral(
                message=message,
                instructions=instructions,
                state=state,
                tool_call_id=tool_call_id,
                delegation_timeout=delegation_timeout,
                parent_llm_getter=parent_llm_getter,
            )

        ephemeral_tool = StructuredTool.from_function(
            coroutine=_delegate_ephemeral_fn,
            name="DelegateEphemeral",
            description=_DELEGATE_EPHEMERAL_DESCRIPTION,
            args_schema=_DelegateEphemeralInput,
            handle_tool_error=True,
        )
        tools.append(ephemeral_tool)

    return tools
