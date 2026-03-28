"""Shared graph-building utilities for agent metaclass and ephemeral agents.

``build_graph`` constructs an uncompiled ``StateGraph`` with the standard
ReAct loop (handler ⇄ ToolNode). It is used by both the ``agent`` metaclass
and the ephemeral delegation tool.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool


def _find_wrap_tool_call(
    middleware: list[Any],
    name: str,
) -> Any | None:
    """Find a single wrap_tool_call callback from middleware list."""
    wrap_tool_call = None
    for mw in middleware:
        wrapper = getattr(mw, "wrap_tool_call", None)
        if callable(wrapper):
            if wrap_tool_call is not None:
                raise ValueError(
                    f"class {name}(agent): multiple middleware provide wrap_tool_call. "
                    f"Only one is supported per agent."
                )
            wrap_tool_call = wrapper
    return wrap_tool_call


def build_graph(
    name: str,
    handler: Any,
    llm: BaseChatModel,
    user_tools: list[BaseTool],
    kit: AgentKit,
    state_type: type = AgentKitState,
    wrap_tool_call: Any | None = None,
) -> Any:
    """Build the ReAct subgraph.

    Returns an uncompiled ``StateGraph``. Call ``.compile()`` on the
    result to get a runnable graph, optionally passing a checkpointer
    for ``interrupt()`` support.
    """
    node_name = name
    all_tools: list[BaseTool] = list(user_tools) + kit.tools
    middleware_list = kit._middleware

    # Resolve which params the handler actually accepts (once, at build time)
    handler_params = inspect.signature(handler).parameters

    async def _agent_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        runtime = ToolRuntime(
            state=state,
            context=kwargs.get("context"),
            config=config,
            stream_writer=kwargs.get("stream_writer", lambda _: None),
            tool_call_id=None,
            store=kwargs.get("store"),
        )
        composed_prompt = kit.prompt(state, runtime)
        bound_llm = llm.bind_tools(all_tools) if all_tools else llm

        available = {
            "llm": bound_llm,
            "tools": list(all_tools),
            "prompt": composed_prompt,
            "runtime": runtime,
        }
        inject = {k: v for k, v in available.items() if k in handler_params}

        result = handler(state, **inject)
        if inspect.isawaitable(result):
            result = await result

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"agent.<locals>.{node_name}"

    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)  # type: ignore[type-var]

    # Allow middleware to add nodes (e.g., Router Node) before edges are wired
    if middleware_list:
        for mw in middleware_list:
            modifier = getattr(mw, "graph_modifier", None)
            if callable(modifier):
                workflow = modifier(workflow, node_name)

    has_router = "router" in workflow.nodes

    if all_tools:
        tool_node_kwargs: dict[str, Any] = {}
        if wrap_tool_call is not None:
            tool_node_kwargs["wrap_tool_call"] = wrap_tool_call
        tool_node = ToolNode(all_tools, **tool_node_kwargs)
        workflow.add_node("tools", tool_node)

        def _should_continue(state: dict[str, Any]) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        workflow.set_entry_point(node_name)
        workflow.add_conditional_edges(
            node_name,
            _should_continue,
            {"tools": "tools", END: END},
        )

        if has_router:
            # Team-aware wiring: tools → router → handler (or END)
            workflow.add_edge("tools", "router")
        else:
            # Standard ReAct wiring: tools → handler
            workflow.add_edge("tools", node_name)
    else:
        workflow.set_entry_point(node_name)
        workflow.add_edge(node_name, END)

    return workflow
