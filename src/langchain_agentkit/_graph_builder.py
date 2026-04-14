"""Shared graph-building utilities.

``build_graph`` constructs an uncompiled ``StateGraph`` with the standard
ReAct loop (handler - ToolNode). It is called by ``AgentKit.compile()``.
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
        + [HumanMessage(content=system_reminder)],
    }


def _merge_before_updates_into_result(
    result: dict[str, Any],
    persisted_before: dict[str, Any],
) -> None:
    """Merge before_model updates into the node's final state update.

    ``messages`` is concatenated so both sources reach the ``add_messages``
    reducer (before_model messages first, preserving chronological order).
    All other keys are filled via ``setdefault`` so existing values in
    ``result`` (from handler / after_model) take precedence — before_model
    only fills in unset keys. Mutates ``result`` in place.
    """
    if not persisted_before:
        return
    before_messages = persisted_before.pop("messages", None)
    for k, v in persisted_before.items():
        result.setdefault(k, v)
    if before_messages:
        combined = list(before_messages)
        combined.extend(result.get("messages") or [])
        result["messages"] = combined


def _process_before_updates(
    before_updates: list[dict[str, Any]],
    handler_state: dict[str, Any],
    persisted_before: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply before_model updates and track persistence.

    Returns an early-exit state update if any hook requested a ``jump_to``
    routing directive; otherwise returns ``None`` and mutates
    ``handler_state`` / ``persisted_before`` in place.
    """
    for update in before_updates:
        if "jump_to" in update:
            state_update = {k: v for k, v in update.items() if k != "jump_to"}
            state_update["_agentkit_jump_to"] = update["jump_to"]
            return state_update
        handler_state.update(update)
        for k, v in update.items():
            persisted_before[k] = v
    return None


def _make_runtime(state: dict[str, Any], config: Any, **kwargs: Any) -> ToolRuntime:
    """Build a ToolRuntime from node arguments (shared by all graph nodes)."""
    return ToolRuntime(
        state=state,
        context=kwargs.get("context"),
        config=config,
        stream_writer=kwargs.get("stream_writer", lambda _: None),
        tool_call_id=None,
        store=kwargs.get("store"),
    )


def build_graph(  # noqa: C901
    name: str,
    handler: Any,
    llm: BaseChatModel,
    user_tools: list[BaseTool],
    kit: AgentKit,
    state_type: type = AgentKitState,
) -> Any:
    """Build the ReAct subgraph.

    Returns an uncompiled ``StateGraph``. Call ``.compile()`` on the
    result to get a runnable graph, optionally passing a checkpointer
    for ``interrupt()`` support.
    """
    node_name = name
    all_tools: list[BaseTool] = list(user_tools) + kit.tools
    extensions_list = kit.extensions

    # Build HookRunner from extensions for lifecycle hooks
    hook_runner = HookRunner(extensions_list)

    # Resolve which params the handler actually accepts (once, at build time)
    handler_params = inspect.signature(handler).parameters

    async def _agent_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        runtime = _make_runtime(state, config, **kwargs)

        # --- before_model hooks ---
        # before_model updates apply to handler_state (for the current LLM
        # call) AND persist into graph state at node exit — symmetric with
        # after_model.  The ``messages`` channel is concatenated with
        # handler/after_model output so the ``add_messages`` reducer sees
        # both sources; other keys yield to handler/after_model on collisions.
        before_updates = await hook_runner.run_before("model", state=state, runtime=runtime)
        handler_state = dict(state)
        persisted_before: dict[str, Any] = {}
        jump_update = _process_before_updates(before_updates, handler_state, persisted_before)
        if jump_update is not None:
            return jump_update

        # --- compose prompt and system reminder (single pass) ---
        # NOTE: called per-step intentionally. Extension prompts are dynamic —
        # they render current state (task list, team status). Do NOT hoist this.
        composition = kit.compose(handler_state, runtime)
        composed_prompt = composition.prompt

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
            # Inject the system-reminder HumanMessage inside the handler
            # boundary so wrap_model hooks (e.g. HistoryExtension) do not
            # observe it in their truncation window — the reminder is
            # strictly ephemeral and must not leak into persisted state.
            request = _inject_system_reminder(request, composition.reminder)
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

            _merge_before_updates_into_result(result, persisted_before)

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"agent.<locals>.{node_name}"

    # --- Run lifecycle nodes ---

    has_run_hooks = hook_runner.has_run_hooks

    async def _run_entry_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        """Run before_run hooks once at graph entry."""
        runtime = _make_runtime(state, config, **kwargs)
        updates = await hook_runner.run_before("run", state=state, runtime=runtime)
        result: dict[str, Any] = {}
        for update in updates:
            result.update(update)
        return result

    async def _run_exit_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        """Run after_run hooks once before graph ends."""
        runtime = _make_runtime(state, config, **kwargs)
        updates = await hook_runner.run_after("run", state=state, runtime=runtime)
        result: dict[str, Any] = {}
        for update in updates:
            result.update(update)
        return result

    # --- Build graph ---

    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)  # type: ignore[type-var]

    # Add run lifecycle nodes BEFORE graph_modifier so extensions (e.g.,
    # the team router) can reference ``_run_exit`` as a terminating
    # destination instead of jumping straight to END and skipping
    # after_run cleanup.
    if has_run_hooks:
        workflow.add_node("_run_entry", _run_entry_node)  # type: ignore[type-var]
        workflow.add_node("_run_exit", _run_exit_node)  # type: ignore[type-var]

    # Allow extensions to add nodes (e.g., Router Node) before edges are wired
    if extensions_list:
        for ext in extensions_list:
            modifier = getattr(ext, "graph_modifier", None)
            if callable(modifier):
                workflow = modifier(workflow, node_name)

    has_router = "router" in workflow.nodes

    if all_tools:
        # Build async tool call wrapper that integrates HookRunner
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

            # --- wrap_tool hooks (onion around handler) ---
            result = await hook_runner.run_wrap(
                "tool",
                state=request,
                handler=handler,
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

        terminal = "_run_exit" if has_run_hooks else END
        destinations = {"tools": "tools", terminal: terminal, node_name: node_name}
        workflow.add_conditional_edges(
            node_name,
            _should_continue,
            destinations,  # type: ignore[arg-type]
        )

        if has_router:
            workflow.add_edge("tools", "router")
        else:
            workflow.add_edge("tools", node_name)

    # Common entry/exit wiring
    if has_run_hooks:
        workflow.set_entry_point("_run_entry")
        workflow.add_edge("_run_entry", node_name)
        if not all_tools:
            workflow.add_edge(node_name, "_run_exit")
        workflow.add_edge("_run_exit", END)
    else:
        workflow.set_entry_point(node_name)
        if not all_tools:
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

    Shared by ``AgentsExtension``'s definition-based and dynamic delegation
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
        # Bind tools so the LLM can emit tool calls — required for
        # ephemeral teammates that ship proxy task tools (and any other
        # ephemeral agent that wants tool use).  The framework contract
        # is "handlers bind their own tools"; this is the handler.
        bound = llm.bind_tools(tools) if tools else llm
        response = await bound.ainvoke(msgs)
        return {"messages": [response], "sender": name}

    kit = AgentKit(extensions=[], prompt=prompt, tools=agent_tools, model=llm, name=name)
    graph = kit.compile(_handler)
    compile_kwargs: dict[str, Any] = {}
    if max_turns is not None:
        compile_kwargs["recursion_limit"] = max_turns * 2
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)
