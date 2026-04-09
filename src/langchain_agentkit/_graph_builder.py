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

from langchain_agentkit.hook_runner import HookRunner
from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

    from langchain_agentkit.agent_kit import AgentKit


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


def _inject_system_reminder(
    handler_state: dict[str, Any],
    system_reminder: str,
) -> dict[str, Any]:
    """Append system-reminder as an ephemeral HumanMessage.

    Returns a new state dict with the reminder appended to messages.
    The original state is NOT mutated. The reminder is never persisted —
    it exists only for the current LLM invocation.
    """
    if not system_reminder:
        return handler_state
    from langchain_core.messages import HumanMessage

    return {
        **handler_state,
        "messages": list(handler_state.get("messages", []))
        + [
            HumanMessage(
                content=f"<system-reminder>\n{system_reminder}\n</system-reminder>",
            ),
        ],
    }


def build_graph(  # noqa: C901
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

        # --- before_model hooks ---
        before_updates = await hook_runner.run_before("model", state=state, runtime=runtime)

        # Apply before_model updates ephemerally (affects LLM call, not
        # persisted to graph state).  Check for jump_to routing directive.
        handler_state = dict(state)
        for update in before_updates:
            if "jump_to" in update:
                state_update = {k: v for k, v in update.items() if k != "jump_to"}
                state_update["_agentkit_jump_to"] = update["jump_to"]
                return state_update
            handler_state.update(update)

        # --- compose prompt and system reminder (single pass) ---
        # NOTE: called per-step intentionally. Extension prompts are dynamic —
        # they render current state (task list, team status). Do NOT hoist this.
        prompt_parts, reminder_parts = kit._collect_contributions(handler_state, runtime)
        composed_prompt = "\n\n".join(prompt_parts)
        # Inject ephemeral system-reminder as a HumanMessage (never stored).
        system_reminder = "\n\n".join(reminder_parts) if reminder_parts else ""
        handler_state = _inject_system_reminder(handler_state, system_reminder)

        # NOTE: tool binding is the handler's responsibility. The framework
        # injects the raw ``llm`` and the composed ``tools`` list; the handler
        # decides when and how to call ``llm.bind_tools(tools, ...)``. This
        # gives implementers full control over provider-specific kwargs
        # (``strict``, ``parallel_tool_calls``, ``tool_choice``) and enables
        # dynamic tool filtering per step.
        available = {
            "llm": llm,
            "tools": list(all_tools),
            "prompt": composed_prompt,
            "runtime": runtime,
        }
        inject = {k: v for k, v in available.items() if k in handler_params}

        # --- wrap_model hooks (onion around handler) ---
        # The request passed to wrap hooks is handler_state so hooks can
        # transform messages before the LLM sees them.  _call_handler
        # uses the request (not a closure) so modifications propagate.
        async def _call_handler(request: Any) -> dict[str, Any]:
            result = handler(request, **inject)
            if inspect.isawaitable(result):
                result = await result
            return result  # type: ignore[no-any-return]

        try:
            result = await hook_runner.run_wrap(
                "model", state=handler_state, handler=_call_handler, runtime=runtime
            )
        except Exception as exc:
            await hook_runner.run_on_error(exc, state=state, runtime=runtime)
            raise

        # --- after_model hooks ---
        after_updates = await hook_runner.run_after("model", state=state, runtime=runtime)

        if isinstance(result, dict):
            for update in after_updates:
                if "jump_to" in update:
                    result["_agentkit_jump_to"] = update["jump_to"]
                else:
                    result.update(update)
            # Clear jump_to if no hook set it (reset from any previous step)
            if "_agentkit_jump_to" not in result:
                result["_agentkit_jump_to"] = None

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"agent.<locals>.{node_name}"

    # --- Run lifecycle nodes ---

    has_run_hooks = (
        hook_runner._hooks.get(("before", "run"))
        or hook_runner._hooks.get(("after", "run"))
        or hook_runner._error_hooks
    )

    async def _run_entry_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        """Run before_run hooks once at graph entry."""
        runtime = ToolRuntime(
            state=state,
            context=kwargs.get("context"),
            config=config,
            stream_writer=kwargs.get("stream_writer", lambda _: None),
            tool_call_id=None,
            store=kwargs.get("store"),
        )
        updates = await hook_runner.run_before("run", state=state, runtime=runtime)
        result: dict[str, Any] = {}
        for update in updates:
            result.update(update)
        return result

    async def _run_exit_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        """Run after_run hooks once before graph ends."""
        runtime = ToolRuntime(
            state=state,
            context=kwargs.get("context"),
            config=config,
            stream_writer=kwargs.get("stream_writer", lambda _: None),
            tool_call_id=None,
            store=kwargs.get("store"),
        )
        updates = await hook_runner.run_after("run", state=state, runtime=runtime)
        result: dict[str, Any] = {}
        for update in updates:
            result.update(update)
        return result

    # --- Build graph ---

    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)  # type: ignore[type-var]

    # Allow extensions to add nodes (e.g., Router Node) before edges are wired
    if extensions_list:
        for ext in extensions_list:
            modifier = getattr(ext, "graph_modifier", None)
            if callable(modifier):
                workflow = modifier(workflow, node_name)

    has_router = "router" in workflow.nodes

    # Add run lifecycle nodes if any run hooks are registered
    if has_run_hooks:
        workflow.add_node("_run_entry", _run_entry_node)  # type: ignore[type-var]
        workflow.add_node("_run_exit", _run_exit_node)  # type: ignore[type-var]

    if all_tools:
        # Build async tool call wrapper that integrates HookRunner + legacy wrap_tool_call
        async def _hooked_wrap_tool_call(request: Any, handler: Any) -> Any:
            tool_name = (
                request.tool_call.get("name", "")
                if isinstance(request.tool_call, dict)
                else getattr(request.tool_call, "name", "")
            )

            # --- before_tool hooks ---
            await hook_runner.run_before(
                "tool",
                state=request.state,
                runtime=request.runtime,
                tool_name=tool_name,
            )

            # --- wrap_tool hooks (onion) + legacy wrap_tool_call ---
            async def _inner_handler(req: Any) -> Any:
                if wrap_tool_call is not None:
                    result = wrap_tool_call(req, handler)
                    # wrap_tool_call is sync but may pass through an async
                    # handler whose return value is a coroutine.
                    if inspect.isawaitable(result):
                        result = await result
                    return result
                return await handler(req)

            result = await hook_runner.run_wrap(
                "tool",
                state=request,
                handler=_inner_handler,
                runtime=request.runtime,
                tool_name=tool_name,
            )

            # --- after_tool hooks ---
            await hook_runner.run_after(
                "tool",
                state=request.state,
                runtime=request.runtime,
                tool_name=tool_name,
            )

            return result

        tool_node = ToolNode(all_tools, awrap_tool_call=_hooked_wrap_tool_call)
        workflow.add_node("tools", tool_node)

        def _should_continue(state: dict[str, Any]) -> str:
            # Read jump_to from per-invocation state (safe for concurrent invocations)
            jump = state.get("_agentkit_jump_to")
            if jump == "end":
                return "_run_exit" if has_run_hooks else END
            if jump == "tools":
                return "tools"
            if jump == "model":
                return node_name

            msgs = state.get("messages", [])
            if not msgs:
                return "_run_exit" if has_run_hooks else END
            last = msgs[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "_run_exit" if has_run_hooks else END

        if has_run_hooks:
            # START → _run_entry → agent → [tools ⇄ agent]* → _run_exit → END
            workflow.set_entry_point("_run_entry")
            workflow.add_edge("_run_entry", node_name)
            destinations = {"tools": "tools", "_run_exit": "_run_exit", node_name: node_name}
            workflow.add_conditional_edges(
                node_name,
                _should_continue,
                destinations,  # type: ignore[arg-type]
            )
            workflow.add_edge("_run_exit", END)
        else:
            # START → agent → [tools ⇄ agent]* → END
            workflow.set_entry_point(node_name)
            destinations = {"tools": "tools", END: END, node_name: node_name}
            workflow.add_conditional_edges(
                node_name,
                _should_continue,
                destinations,  # type: ignore[arg-type]
            )

        if has_router:
            workflow.add_edge("tools", "router")
        else:
            workflow.add_edge("tools", node_name)
    else:
        if has_run_hooks:
            workflow.set_entry_point("_run_entry")
            workflow.add_edge("_run_entry", node_name)
            workflow.add_edge(node_name, "_run_exit")
            workflow.add_edge("_run_exit", END)
        else:
            workflow.set_entry_point(node_name)
            workflow.add_edge(node_name, END)

    return workflow


def build_ephemeral_graph(
    name: str,
    llm: Any,
    prompt: str,
    *,
    user_tools: list[BaseTool] | None = None,
    max_turns: int | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile a minimal ReAct graph for ephemeral / config-based agents.

    Shared by ``AgentExtension``'s definition-based and dynamic delegation
    paths and by ``TeamExtension``'s ephemeral teammates. The handler simply
    prepends a ``SystemMessage`` carrying ``prompt`` and invokes ``llm``.

    Args:
        name: Graph name; also used as the ``sender`` on produced messages.
        llm: Chat model used for the single model call.
        prompt: System prompt injected on every turn.
        user_tools: Optional list of tools the agent can invoke.
        max_turns: When set, caps ``recursion_limit`` to ``max_turns * 2``.
        checkpointer: Optional checkpointer forwarded to ``compile``.
    """
    from langchain_agentkit.agent_kit import AgentKit

    agent_tools = list(user_tools or [])

    async def _handler(
        handler_state: dict[str, Any],
        *,
        llm: Any,
        prompt: str,
        tools: Any = None,
    ) -> dict[str, Any]:
        from langchain_core.messages import SystemMessage

        msgs = [SystemMessage(content=prompt)] + list(handler_state.get("messages", []))
        response = await llm.ainvoke(msgs)
        return {"messages": [response], "sender": name}

    kit = AgentKit(extensions=[], prompt=prompt)
    graph = build_graph(
        name=name,
        handler=_handler,
        llm=llm,
        user_tools=agent_tools,
        kit=kit,
        state_type=AgentKitState,
    )
    compile_kwargs: dict[str, Any] = {}
    if max_turns is not None:
        compile_kwargs["recursion_limit"] = max_turns * 2
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)
