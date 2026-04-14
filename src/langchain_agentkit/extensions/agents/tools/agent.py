"""Unified Agent tool for LangGraph agent delegation.

Tools return ``Command(update={"messages": [ToolMessage(...)]})`` to
propagate delegation results. Compatible with LangGraph's ``ToolNode``
out of the box.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.graph import END
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_agentkit._graph_builder import build_ephemeral_graph
from langchain_agentkit.extensions.agents.refs import (
    Dynamic,
    Predefined,
    resolve_agent_by_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool


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


def _compile_or_resolve(
    target: Any,
    compiled_cache: dict[str, Any],
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
) -> Any:
    """Resolve a delegation target to an invocable object."""
    from langchain_agentkit.composability import AgentLike

    if isinstance(target, AgentLike):
        return target

    return _compile_agent(target, compiled_cache, parent_tools_getter)


def _compile_agent(
    graph: Any,
    compiled_cache: dict[str, Any],
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
) -> Any:
    """Compile a raw StateGraph agent, caching the result."""
    name: str = graph.name
    if name in compiled_cache:
        return compiled_cache[name]

    if getattr(graph, "tools_inherit", False) and parent_tools_getter is not None:
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
                    lambda state: (
                        "tools"
                        if (
                            state["messages"][-1].tool_calls
                            if hasattr(state["messages"][-1], "tool_calls")
                            else False
                        )
                        else END
                    ),
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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Delegation to '{agent}' timed out after {timeout}s",
                        tool_call_id=tool_call_id,
                    ),
                ],
            }
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Delegation to '%s' failed",
            agent,
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Delegation to '{agent}' failed due to an internal error.",
                        tool_call_id=tool_call_id,
                    ),
                ],
            }
        )

    response = _extract_final_response(result)
    return Command(
        update={
            "messages": [
                ToolMessage(content=response, tool_call_id=tool_call_id),
            ],
        }
    )


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
    model_resolver: Callable[[str], Any] | None = None,
    skills_resolver: Callable[[list[str]], str] | None = None,
) -> Any:
    """Delegate a task to a pre-defined or dynamically created agent."""
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
            parent_llm_getter=parent_llm_getter,
            model_resolver=model_resolver,
            skills_resolver=skills_resolver,
        )

    if "prompt" in agent_ref:
        if not ephemeral:
            raise ToolException(
                "Dynamic agents are not enabled. "
                "Set ephemeral=True on AgentsExtension to allow custom agents."
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
    parent_llm_getter: Callable[[], Any] | None = None,
    model_resolver: Callable[[str], Any] | None = None,
    skills_resolver: Callable[[list[str]], str] | None = None,
) -> Any:
    """Delegate a task to a named pre-defined agent."""
    target = resolve_agent_by_name(agent_id, agents_by_name)

    from langchain_agentkit.extensions.agents.types import AgentConfig

    agent_config = getattr(target, "_agent_config", None)
    if isinstance(agent_config, AgentConfig):
        return await _delegate_agent_config(
            agent_config=agent_config,
            message=message,
            delegation_timeout=delegation_timeout,
            parent_llm_getter=parent_llm_getter,
            parent_tools_getter=parent_tools_getter,
            model_resolver=model_resolver,
            skills_resolver=skills_resolver,
            tool_call_id=tool_call_id,
        )

    compiled = _compile_or_resolve(target, compiled_cache, parent_tools_getter)
    scoped_state = _build_scoped_state(message)

    return await _run_delegation(
        compiled=compiled,
        scoped_state=scoped_state,
        agent=agent_id,
        timeout=delegation_timeout,
        tool_call_id=tool_call_id,
    )


async def _delegate_agent_config(
    agent_config: Any,
    message: str,
    delegation_timeout: float,
    parent_llm_getter: Callable[[], Any] | None,
    parent_tools_getter: Callable[[], list[BaseTool]] | None,
    model_resolver: Callable[[str], Any] | None,
    skills_resolver: Callable[[list[str]], str] | None,
    tool_call_id: str,
) -> Any:
    """Delegate to an AgentConfig — resolve model, tools, skills, max_turns."""
    if agent_config.model and model_resolver:
        llm = model_resolver(agent_config.model)
    elif parent_llm_getter is not None:
        llm = parent_llm_getter()
    else:
        raise ToolException(
            "Agent definition requires a model but no parent LLM or model_resolver is available."
        )

    agent_tools: list[BaseTool] = []
    if agent_config.tools is not None and parent_tools_getter is not None:
        parent_tools = parent_tools_getter()
        allowed = set(agent_config.tools)
        agent_tools = [t for t in parent_tools if t.name in allowed]

    prompt = agent_config.prompt
    if agent_config.skills and skills_resolver:
        skill_content = skills_resolver(agent_config.skills)
        if skill_content:
            prompt = prompt + "\n\n" + skill_content

    agent_name = agent_config.name

    try:
        compiled = build_ephemeral_graph(
            name=agent_name,
            llm=llm,
            prompt=prompt,
            user_tools=agent_tools,
            max_turns=agent_config.max_turns,
        )
    except Exception as exc:
        raise ToolException(f"Failed to create agent '{agent_name}': {exc}") from exc

    scoped_state = _build_scoped_state(message)

    return await _run_delegation(
        compiled=compiled,
        scoped_state=scoped_state,
        agent=agent_name,
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

    try:
        compiled = build_ephemeral_graph(
            name=ephemeral_name,
            llm=llm,
            prompt=prompt,
        )
    except Exception as exc:
        raise ToolException(f"Failed to create dynamic agent: {exc}") from exc

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


_AGENT_DESCRIPTION = """Delegate a task to an agent and wait for the result.

Use when:
- The task needs specialized tools or knowledge you don't have directly
- The task is independent and can be done in isolation from your current context
- You need several things done in parallel (call Agent multiple times in one turn)
- The task benefits from focused, context-isolated execution

Do NOT use when:
- The task requires your full conversation context
- The task is trivial (faster to do yourself)
- The task requires back-and-forth discussion

## Writing the prompt

Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't know why this task matters.
- Explain what you're trying to accomplish and why.
- Describe what you've already learned or ruled out.
- Give enough context that the agent can make judgment calls rather than following a narrow instruction.
- If you need a short response, say so (e.g., "summarize in under 200 words").

**Never delegate understanding.** Don't write "based on your findings, decide what to do" or "based on the research, produce the final summary." Those phrases push synthesis onto the agent. Write prompts that prove you understood: include the specific facts, sources, or identifiers the agent should use.

The agent receives ONLY your message — it has no access to your conversation history.

## Parallel launches

When subtasks are independent, issue multiple Agent calls in the same turn. Each agent runs in isolation, so parallel launches are the fastest way to fan out work.

## Examples

<example>
user: "Compare three candidate vendors for our new CRM"
assistant: I'll delegate research to three agents in parallel — one per vendor.
Agent(agent={id: "researcher"}, message="Research vendor A: pricing, integrations, support reputation, data residency. Summarize strengths and weaknesses in under 200 words.")
Agent(agent={id: "researcher"}, message="Research vendor B: ...")
Agent(agent={id: "researcher"}, message="Research vendor C: ...")
<commentary>
Three independent research tasks in parallel. Each agent gets a focused prompt. The lead synthesizes results after all three return.
</commentary>
</example>

<example>
user: "Summarize the key points of this 40-page contract"
assistant: I've skimmed it; the material sections are payment (§4), termination (§9), IP assignment (§12).
[reads those sections directly and summarizes]
<commentary>
The assistant does NOT delegate — it already has the document in context. Delegating would push understanding onto the agent.
</commentary>
</example>"""


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
    model_resolver: Callable[[str], Any] | None = None,
    skills_resolver: Callable[[list[str]], str] | None = None,
) -> list[BaseTool]:
    """Create the unified Agent tool for agent-to-agent delegation."""
    if ephemeral and parent_llm_getter is None:
        msg = "parent_llm_getter is required when ephemeral=True"
        raise ValueError(msg)

    schema = _AgentDynamicInput if ephemeral else _AgentInput

    # Use a real async function (not ``functools.partial``) so LangGraph's
    # ``ToolNode._get_all_injected_args`` can introspect it via
    # ``typing.get_type_hints``. ``get_type_hints`` rejects ``functools.partial``
    # with ``TypeError: is not a module, class, method, or function.``
    #
    # We intentionally do NOT use ``functools.wraps(_agent_tool)`` here: that
    # would copy ``_agent_tool``'s ``__annotations__`` (which contain
    # ``Callable[...]`` forward refs that aren't resolvable at runtime under
    # ``from __future__ import annotations``) and break ``get_type_hints``
    # with ``NameError: name 'Callable' is not defined``.
    async def bound(**llm_kwargs: Any) -> Any:
        return await _agent_tool(
            agents_by_name=agents_by_name,
            compiled_cache=compiled_cache,
            delegation_timeout=delegation_timeout,
            parent_tools_getter=parent_tools_getter,
            ephemeral=ephemeral,
            parent_llm_getter=parent_llm_getter,
            model_resolver=model_resolver,
            skills_resolver=skills_resolver,
            **llm_kwargs,
        )

    tool = StructuredTool.from_function(
        coroutine=bound,
        name="Agent",
        description=_AGENT_DESCRIPTION,
        args_schema=schema,
        handle_tool_error=True,
    )

    return [tool]
