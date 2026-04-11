"""Metaclass-driven LangGraph agent node with extension support.

Usage::

    from langchain_agentkit import agent

    class researcher(agent):
        model = ChatOpenAI(model="gpt-4o")
        tools = [web_search]
        extensions = [SkillsExtension(skills="skills/"), TasksExtension()]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, tools, prompt, runtime):
            bound = llm.bind_tools(tools)
            response = await bound.ainvoke(state["messages"])
            return {"messages": [response], "sender": "researcher"}

``researcher`` is a ``StateGraph`` — call ``.compile()`` to get a
runnable graph, optionally passing a checkpointer for ``interrupt()``
support.

The ``model`` attribute accepts either a ``BaseChatModel`` instance or
a string. Strings are resolved at build time via ``AgentKit.model_resolver``.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import TYPE_CHECKING, Any

from langchain_agentkit._graph_builder import _find_wrap_tool_call, build_graph
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.state import AgentKitState


def _run_async_setup(kit: AgentKit) -> None:
    """Run kit.asetup() synchronously, handling running event loops."""
    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False

    if has_loop:
        exc: BaseException | None = None

        def _run() -> None:
            nonlocal exc
            try:
                asyncio.run(kit.asetup())
            except BaseException as e:
                exc = e

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        if exc is not None:
            raise exc
    else:
        asyncio.run(kit.asetup())


if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

    from langchain_agentkit.extension import Extension


def _validate_handler_signature(
    handler: Any,
    class_name: str,
    valid_params: frozenset[str],
    label: str,
) -> type:
    """Validate handler signature and extract state type.

    Returns the state type inferred from the handler's first parameter
    annotation. If no annotation is present, defaults to ``AgentKitState``.
    """
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if not params:
        raise ValueError(
            f"class {class_name}({label}): handler must accept at least "
            f"'state' as its first parameter"
        )

    first = params[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(
            f"class {class_name}({label}): handler's first parameter must be "
            f"positional ('state'), got {first.kind.name}"
        )

    state_type: type = AgentKitState
    if first.annotation is not inspect.Parameter.empty:
        state_type = first.annotation

    for param in params[1:]:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            if param.name not in valid_params:
                raise ValueError(
                    f"class {class_name}({label}): unknown handler parameter "
                    f"'{param.name}'. Valid parameters: state, "
                    f"{', '.join(sorted(valid_params))}"
                )
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"class {class_name}({label}): handler parameter '{param.name}' "
                f"must be keyword-only (after *). "
                f"Signature should be: handler(state, *, {param.name}, ...)"
            )

    return state_type


# Valid injectable parameter names for handler (besides 'state')
_INJECTABLE_PARAMS = frozenset({"llm", "tools", "prompt", "runtime"})


class _AgentMeta(type):
    """Metaclass that intercepts class body and returns a StateGraph.

    When a class inherits from ``agent``, this metaclass:

    1. Extracts ``model``, ``tools``, ``extensions``, ``prompt``, ``handler``,
       ``skills``, ``max_turns`` from the class body.
    2. Validates the handler signature and infers state type from annotation.
    3. Builds an ``AgentKit`` from extensions and prompt source.
    4. Resolves model via ``AgentKit.resolve_model()`` if it's a string.
    5. Merges user tools with kit tools.
    6. Builds and returns an uncompiled ``StateGraph`` with the ReAct loop.

    The result is NOT a class — it's a ``StateGraph``. Call ``.compile()``
    to get a runnable graph, optionally passing a checkpointer for
    ``interrupt()`` support.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        # The `agent` base class itself — create normally
        if not bases:
            return super().__new__(mcs, name, bases, namespace)

        # Subclass of agent → intercept and return StateGraph

        # Extract handler (required)
        handler = namespace.get("handler")
        if handler is None:
            raise ValueError(f"class {name}(agent) must define an async def handler(...) function")
        if not callable(handler):
            raise ValueError(
                f"class {name}(agent): handler must be callable, got {type(handler).__name__}"
            )

        # Validate handler signature and extract state type
        state_type = _validate_handler_signature(handler, name, _INJECTABLE_PARAMS, "agent")

        # Extract model (required) — string or BaseChatModel
        model_raw = namespace.get("model")
        if model_raw is None:
            raise ValueError(
                f"class {name}(agent) must define a model attribute "
                f"(e.g. model = ChatOpenAI(model='gpt-4o') or model = 'gpt-4o')"
            )

        # Extract tools — supports list, "inherit" sentinel, or absent (no tools)
        user_tools_raw = namespace.get("tools", [])
        tools_inherit = False
        if isinstance(user_tools_raw, str):
            if user_tools_raw != "inherit":
                raise ValueError(
                    f"class {name}(agent): tools must be a list or 'inherit', "
                    f"got string '{user_tools_raw}'"
                )
            user_tools: list[BaseTool] = []
            tools_inherit = True
        elif isinstance(user_tools_raw, (list, tuple)):
            user_tools = list(user_tools_raw)
        else:
            raise ValueError(
                f"class {name}(agent): tools must be a list or 'inherit', "
                f"got {type(user_tools_raw).__name__}"
            )

        extensions: list[Extension] = namespace.get("extensions", [])
        if not isinstance(extensions, (list, tuple)):
            raise ValueError(
                f"class {name}(agent): extensions must be a list, got {type(extensions).__name__}"
            )

        prompt_source: str | Path | list[str | Path] | None = namespace.get("prompt")
        description: str = namespace.get("description", "")
        skills: list[str] = namespace.get("skills", [])
        max_turns: int | None = namespace.get("max_turns")

        kit = AgentKit(extensions=list(extensions), prompt=prompt_source)

        # Run async setup (discovery, cross-extension wiring)
        _run_async_setup(kit)

        # Resolve model — string goes through model_resolver, object used as-is
        llm = kit.resolve_model(model_raw)

        # Use composed schema from extensions unless handler explicitly annotates state
        if state_type is AgentKitState:
            state_type = kit.state_schema

        wrap_tool_call = _find_wrap_tool_call(extensions, name)

        graph = build_graph(
            name=name,
            handler=handler,
            llm=llm,
            user_tools=list(user_tools),
            kit=kit,
            state_type=state_type,
            wrap_tool_call=wrap_tool_call,
        )

        # Attach metadata
        graph.name = name
        graph.description = description
        graph.tools_inherit = tools_inherit
        graph.skills = skills
        graph.max_turns = max_turns

        # Store rebuild ingredients for team proxy tool injection
        graph._agentkit_handler = handler
        graph._agentkit_llm = llm
        graph._agentkit_user_tools = list(user_tools)
        graph._agentkit_kit = kit

        return graph


class agent(metaclass=_AgentMeta):  # noqa: N801
    """Base class for extension-aware LangGraph agent nodes.

    Declare a subclass to create a ``StateGraph`` with an automatic
    ReAct loop (handler - ToolNode) and extension-composed tools and
    prompts. Call ``.compile()`` on the result to get a runnable graph.

    Example::

        from langchain_agentkit import agent

        class researcher(agent):
            model = ChatOpenAI(model="gpt-4o")
            tools = [web_search]
            extensions = [SkillsExtension(skills="skills/"), TasksExtension()]
            prompt = "You are a research assistant."

            async def handler(state, *, llm, tools, prompt, runtime):
                bound = llm.bind_tools(tools)
                response = await bound.ainvoke(state["messages"])
                return {"messages": [response], "sender": "researcher"}

        # Model can be a string (resolved via model_resolver):
        class fast_agent(agent):
            model = "gpt-4o-mini"
            extensions = [SkillsExtension(skills="skills/")]
            ...

    Class attributes:

        model: Required. A BaseChatModel instance or a string resolved
            via ``AgentKit.model_resolver``.
        tools: Optional. Agent-specific tools (not from extensions).
        extensions: Optional. Ordered list of Extension instances.
        prompt: Optional. System prompt template — inline string, file path,
            or list of either.
        description: Optional. Used for delegation matching.
        skills: Optional. List of skill names to preload into the prompt.
        max_turns: Optional. Maximum agentic turns before the agent stops.

    Handler signature::

        async def handler(state, *, llm, tools, prompt, runtime): ...

    ``state`` is positional. Everything after ``*`` is keyword-only and
    injected by name — declare only what you need, in any order.

    Injectable parameters:

        llm: The raw model, exactly as declared in the ``model`` attribute.
            Tool binding is the handler's responsibility — call
            ``llm.bind_tools(tools, ...)`` when you need tool calling. This
            gives implementers full control over provider-specific kwargs
            (``strict``, ``parallel_tool_calls``, ``tool_choice``) and
            enables dynamic tool filtering per step.
        tools: Complete tool list (user tools + extension tools).
        prompt: Fully composed system prompt (template + extension sections).
        runtime: ToolRuntime — unified runtime context. Use
            ``runtime.config`` for the full ``RunnableConfig``.
    """
