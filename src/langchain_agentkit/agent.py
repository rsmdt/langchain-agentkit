"""Declarative LangGraph agent with extension support.

**Static** — all properties are class attributes::

    from langchain_agentkit import Agent

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension(skills="skills/")]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, tools, prompt, runtime):
            ...

    graph = Researcher().compile()

**Dynamic** — properties resolved per-request via sync/async methods::

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")

        async def prompt(self):
            result = await self.backend.read(".agentkit/AGENTS.md")
            return result.content or ""

        def extensions(self):
            return [FilesystemExtension(backend=self.backend)]

        async def handler(state, *, llm, tools, prompt, runtime):
            ...

    graph = Researcher(backend=DaytonaBackend(sandbox)).compile()
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import TYPE_CHECKING, Any

from langchain_agentkit.agent_kit import AgentKit, run_extension_setup
from langchain_agentkit.state import AgentKitState

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

        # Validate handler signature (state type is resolved by kit.state_schema)
        _validate_handler_signature(handler, name, _INJECTABLE_PARAMS, "agent")

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

        graph = _build_agent_graph(
            handler=handler,
            model=model_raw,
            extensions=list(extensions),
            tools=list(user_tools),
            prompt=prompt_source,
            name=name,
        )

        # Attach metaclass-specific metadata
        graph.description = description
        graph.tools_inherit = tools_inherit
        graph.skills = skills
        graph.max_turns = max_turns

        return graph


class agent(metaclass=_AgentMeta):  # noqa: N801
    """Legacy metaclass that intercepts class body and returns a StateGraph.

    Prefer the ``Agent`` base class for new code. The metaclass is retained
    for backward compatibility.
    """


# Properties that are callables but must NOT be called by _resolve()
# because they take arguments (model_resolver) or are the handler itself.
_NO_CALL = frozenset({"handler", "model_resolver"})


def _build_agent_graph(
    *,
    handler: Any,
    model: Any,
    extensions: list[Extension],
    tools: list[BaseTool],
    prompt: str | Path | list[str | Path] | None = None,
    model_resolver: Any = None,
    name: str = "agent",
    stream_tool_results: bool = True,
) -> Any:
    """Shared graph-building pipeline for both Agent and legacy metaclass.

    Creates an AgentKit, runs async setup, compiles the graph, and
    attaches rebuild metadata.
    """
    kit = AgentKit(
        extensions=extensions,
        prompt=prompt,
        tools=tools,
        model=model,
        model_resolver=model_resolver,
        name=name,
        stream_tool_results=stream_tool_results,
    )

    _run_coroutine(run_extension_setup(kit))

    # Resolve model string → BaseChatModel once, store back so
    # kit.compile() doesn't re-resolve through model_resolver.
    resolved_model = kit.model
    kit._model_raw = resolved_model

    state_graph = kit.compile(handler)

    # Attach metadata for delegation and team proxy rebuilds
    state_graph.name = name
    state_graph._agentkit_handler = handler
    state_graph._agentkit_llm = resolved_model
    state_graph._agentkit_user_tools = list(tools)
    state_graph._agentkit_kit = kit

    return state_graph


def _run_coroutine(coro: Any) -> Any:
    """Run an awaitable synchronously, handling running event loops.

    Bridges sync and async contexts, spawning a thread when needed.
    """
    if not inspect.isawaitable(coro):
        return coro

    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False

    if has_loop:
        result_box: list[Any] = [None]
        exc_box: list[BaseException | None] = [None]

        def _run() -> None:
            try:
                result_box[0] = asyncio.run(coro)  # type: ignore[arg-type]
            except BaseException as e:
                exc_box[0] = e

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        if exc_box[0] is not None:
            raise exc_box[0]
        return result_box[0]

    return asyncio.run(coro)  # type: ignore[arg-type]


class Agent:
    """Declarative agent with flexible property resolution.

    Each configurable property (``model``, ``prompt``, ``extensions``,
    ``tools``, ``model_resolver``) can be declared as:

    1. **Static class attribute** — ``model = ChatOpenAI(model="gpt-4o")``
    2. **Instance attribute** — set via ``__init__`` kwargs
    3. **Sync method** — ``def extensions(self): return [...]``
    4. **Async method** — ``async def prompt(self): return await ...``

    ``handler`` is always a function defined on the class (no ``self``),
    passed directly to ``kit.compile(handler)``.

    Two entry points:

    - ``compile(**kwargs)`` — returns a compiled, invocable graph.
      Pass ``checkpointer``, ``recursion_limit``, etc.
    - ``graph()`` — returns an uncompiled ``StateGraph`` for
      composition (delegation targets, team members, subgraphs).

    Example — fully static::

        class Researcher(Agent):
            model = ChatOpenAI(model="gpt-4o")
            prompt = "You are a researcher."
            extensions = [SkillsExtension(skills="skills/")]

            async def handler(state, *, llm, tools, prompt, runtime):
                ...

        app = Researcher().compile()

    Example — dynamic backend::

        class Researcher(Agent):
            model = ChatOpenAI(model="gpt-4o")

            async def prompt(self):
                result = await self.backend.read(".agentkit/AGENTS.md")
                return result.content or ""

            def extensions(self):
                return [FilesystemExtension(backend=self.backend)]

            async def handler(state, *, llm, tools, prompt, runtime):
                ...

        app = Researcher(backend=DaytonaBackend(sandbox)).compile()
    """

    model: Any = None
    model_resolver: Any = None
    prompt: Any = None
    extensions: Any = []
    tools: Any = []
    stream_tool_results: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def _resolve(self, name: str) -> Any:
        """Resolve a property through the static/method/async chain.

        Properties in ``_NO_CALL`` (handler, model_resolver) are returned
        as-is without calling — they are callables that take arguments.
        """
        val = getattr(self, name)
        if name in _NO_CALL:
            return val
        if callable(val):
            val = val()
        if inspect.isawaitable(val):
            val = await val
        return val

    def graph(self) -> Any:
        """Resolve all properties and build an uncompiled ``StateGraph``.

        Use this when you need the raw graph for composition — passing
        to ``AgentsExtension(agents=[...])``, ``TeamExtension(agents=[...])``,
        or embedding as a subgraph in a parent workflow.

        Returns:
            An uncompiled ``StateGraph``.
        """
        model = _run_coroutine(self._resolve("model"))
        prompt = _run_coroutine(self._resolve("prompt"))
        extensions = _run_coroutine(self._resolve("extensions"))
        tools = _run_coroutine(self._resolve("tools"))
        model_resolver = _run_coroutine(self._resolve("model_resolver"))

        # Get handler as raw function, not bound method. When defined
        # in a class body as `async def handler(state, *, llm): ...`,
        # `self.handler` would be a bound method injecting `self` as
        # the first arg. We need the unbound function.
        handler = type(self).__dict__.get("handler")
        if handler is None:
            handler = getattr(self, "handler", None)
        if handler is None:
            raise ValueError(f"{type(self).__name__} must define a handler function")

        cls = type(self)
        state_graph = _build_agent_graph(
            handler=handler,
            model=model,
            model_resolver=model_resolver,
            extensions=list(extensions) if extensions else [],
            tools=list(tools) if tools else [],
            prompt=prompt,
            name=getattr(self, "name", cls.__name__),
            stream_tool_results=bool(getattr(self, "stream_tool_results", True)),
        )

        # Attach Agent-specific metadata
        state_graph.description = getattr(self, "description", "")
        state_graph.tools_inherit = getattr(self, "tools_inherit", False)
        state_graph.skills = getattr(self, "skills", [])
        state_graph.max_turns = getattr(self, "max_turns", None)

        return state_graph

    def compile(self, **kwargs: Any) -> Any:
        """Build and compile in one step — returns a runnable graph.

        Shorthand for ``self.graph().compile(**kwargs)``. Pass
        ``checkpointer``, ``recursion_limit``, etc. as keyword arguments.

        When ``stream_tool_results=False`` (or any extension suppresses a
        tool via :meth:`Extension.stream_tool_results`), the returned
        runnable is transparently wrapped in
        :class:`~langchain_agentkit.streaming.FilteredGraph` so ``astream``
        and ``astream_events`` redact tool-result payloads on the outbound
        stream. When no tool is suppressed, the raw compiled graph is
        returned unchanged.

        Returns:
            A compiled, invocable graph — possibly a
            :class:`~langchain_agentkit.streaming.FilteredGraph` proxy.
        """
        from langchain_agentkit.streaming import wrap_if_filtering

        state_graph = self.graph()
        compiled = state_graph.compile(**kwargs)
        kit = getattr(state_graph, "_agentkit_kit", None)
        suppressed = kit.suppressed_tool_names() if kit is not None else frozenset()
        return wrap_if_filtering(compiled, suppressed)
