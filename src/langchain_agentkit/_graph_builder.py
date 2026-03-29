"""Shared graph-building utilities for agent metaclass and ephemeral agents.

``build_graph`` constructs an uncompiled ``StateGraph`` with the standard
ReAct loop (handler - ToolNode). It is used by both the ``agent`` metaclass
and the ephemeral delegation tool.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.hook_runner import HookRunner
from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool


def _find_wrap_tool_call(
    extensions: list[Any],
    name: str,
) -> Any | None:
    """Find a single wrap_tool_call callback from extensions list.

    Note: This is the legacy single-callback mechanism. The HookRunner's
    wrap("tool") hooks provide the newer onion-style composition.
    """
    wrap_tool_call = None
    for ext in extensions:
        wrapper = getattr(ext, "wrap_tool_call", None)
        if callable(wrapper):
            if wrap_tool_call is not None:
                raise ValueError(
                    f"class {name}(agent): multiple extensions provide wrap_tool_call. "
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
    extensions_list = kit._extensions

    # Build HookRunner from extensions for lifecycle hooks
    hook_runner = HookRunner(extensions_list)

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

        # --- before_run hooks (first invocation only) ---
        # Note: before_run/after_run are run externally by the caller
        # who manages the full run lifecycle, not per-step here.

        # --- before_model hooks ---
        before_updates = await hook_runner.run_before("model", state=state, runtime=runtime)

        # Check for jump_to in before_model results
        for update in before_updates:
            if "jump_to" in update:
                # Return the update with jump_to for the routing function to handle
                return update

        # --- process_history ---
        messages = state.get("messages", [])
        messages = hook_runner.run_process_history(list(messages))

        # --- compose prompt and bind tools ---
        composed_prompt = kit.prompt(state, runtime)
        bound_llm = llm.bind_tools(all_tools) if all_tools else llm

        available = {
            "llm": bound_llm,
            "tools": list(all_tools),
            "prompt": composed_prompt,
            "runtime": runtime,
        }
        inject = {k: v for k, v in available.items() if k in handler_params}

        # --- wrap_model hooks (onion around handler) ---
        async def _call_handler(request: Any) -> dict[str, Any]:
            result = handler(state, **inject)
            if inspect.isawaitable(result):
                result = await result
            return result  # type: ignore[no-any-return]

        try:
            result = await hook_runner.run_wrap("model", request=state, handler=_call_handler)
        except Exception as exc:
            await hook_runner.run_on_error(exc, state=state, runtime=runtime)
            raise

        # --- after_model hooks ---
        after_updates = await hook_runner.run_after("model", state=state, runtime=runtime)

        # Merge after_model updates into result if they contain state changes
        if isinstance(result, dict):
            for update in after_updates:
                if "jump_to" in update:
                    # Propagate jump_to through the result
                    result["jump_to"] = update["jump_to"]
                else:
                    result.update(update)

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"agent.<locals>.{node_name}"

    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)  # type: ignore[type-var]

    # Allow extensions to add nodes (e.g., Router Node) before edges are wired
    if extensions_list:
        for ext in extensions_list:
            modifier = getattr(ext, "graph_modifier", None)
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
            # Check for jump_to routing from hooks
            jump_to = state.get("jump_to")
            if jump_to == "end":
                return END
            if jump_to == "tools":
                return "tools"

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
            workflow.add_edge("tools", "router")
        else:
            workflow.add_edge("tools", node_name)
    else:
        workflow.set_entry_point(node_name)
        workflow.add_edge(node_name, END)

    return workflow
