"""Declarative LangGraph agent with extension support.

``Agent.graph()`` and ``Agent.compile()`` are async because property
resolution and extension setup may issue backend I/O. Callers must
``await`` them.

**Static** — all properties are class attributes::

    from langchain_agentkit import Agent

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension(skills="skills/")]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, tools, prompt, runtime):
            ...

    graph = await Researcher().compile()

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

    graph = await Researcher(backend=DaytonaBackend(sandbox)).compile()
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from langchain_agentkit.agent_kit import AgentKit, run_extension_setup

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

    from langchain_agentkit.extension import Extension


# Properties that are callables but must NOT be called by _resolve()
# because they take arguments (model_resolver) or are the handler itself.
_NO_CALL = frozenset({"handler", "model_resolver"})


async def _build_agent_graph(
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
    """Build the ReAct graph for an :class:`Agent`.

    Creates an :class:`AgentKit`, awaits async extension setup,
    compiles the graph, and attaches rebuild metadata. Async because
    extension setup may issue backend I/O (skill discovery, env probe,
    etc.) and because backends may hold loop-bound resources — running
    setup in a worker thread with a fresh loop would break those
    bindings.
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

    await run_extension_setup(kit)

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

    Two async entry points (callers must ``await``):

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

        app = await Researcher().compile()

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

        app = await Researcher(backend=DaytonaBackend(sandbox)).compile()
    """

    model: Any = None
    model_resolver: Any = None
    prompt: Any = None
    extensions: Any = ()
    # ``tools`` defaults to ``"inherit"`` so sub-agents borrow their parent's
    # toolset by default at delegation time. Override with an explicit list to
    # supply a fixed toolset; include the literal ``"inherit"`` sentinel in
    # that list to add tools on top of the inherited set instead of replacing
    # it. Examples:
    #   tools = "inherit"            # parent's tools (default)
    #   tools = []                   # no tools, even if delegated to
    #   tools = [my_tool]            # just my_tool, no inherit
    #   tools = [my_tool, "inherit"] # my_tool + parent's tools at delegation
    tools: Any = "inherit"
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

    async def graph(self) -> Any:
        """Resolve all properties and build an uncompiled ``StateGraph``.

        Async because property resolution may await async methods
        (``async def prompt(self)``, ``async def model(self)``, …) and
        because extension setup awaits backend I/O. Callers must
        ``await``::

            state_graph = await Researcher().graph()

        Use this when you need the raw graph for composition — passing
        to ``AgentsExtension(agents=[...])``, ``TeamExtension(agents=[...])``,
        or embedding as a subgraph in a parent workflow.

        Returns:
            An uncompiled ``StateGraph``.
        """
        model = await self._resolve("model")
        prompt = await self._resolve("prompt")
        extensions = await self._resolve("extensions")
        raw_tools = await self._resolve("tools")
        model_resolver = await self._resolve("model_resolver")

        # Parse ``tools``: detect the ``"inherit"`` sentinel and split the
        # value into a clean tool list + the ``tools_inherit`` flag the
        # delegation runtime reads. Bare ``"inherit"`` (default) is treated
        # as ``["inherit"]``. ``None`` is normalized to ``[]``.
        tools_seq: list[Any]
        if raw_tools is None:
            tools_seq = []
        elif isinstance(raw_tools, str):
            tools_seq = [raw_tools]
        else:
            tools_seq = list(raw_tools)
        tools_inherit = "inherit" in tools_seq
        clean_tools = [t for t in tools_seq if t != "inherit"]

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
        state_graph = await _build_agent_graph(
            handler=handler,
            model=model,
            model_resolver=model_resolver,
            extensions=list(extensions) if extensions else [],
            tools=clean_tools,
            prompt=prompt,
            name=getattr(self, "name", cls.__name__),
            stream_tool_results=bool(getattr(self, "stream_tool_results", True)),
        )

        # Attach Agent-specific metadata
        state_graph.description = getattr(self, "description", "")
        state_graph.tools_inherit = tools_inherit
        state_graph.skills = getattr(self, "skills", [])
        state_graph.max_turns = getattr(self, "max_turns", None)

        return state_graph

    async def compile(self, **kwargs: Any) -> Any:
        """Build and compile in one step — returns a runnable graph.

        Async shorthand for ``(await self.graph()).compile(**kwargs)``.
        Pass ``checkpointer``, ``recursion_limit``, etc. as keyword
        arguments. Callers must ``await``::

            graph = await Researcher(backend=backend).compile()
            result = await graph.ainvoke({"messages": [...]})

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

        state_graph = await self.graph()
        compiled = state_graph.compile(**kwargs)
        kit = getattr(state_graph, "_agentkit_kit", None)
        suppressed = kit.suppressed_tool_names() if kit is not None else frozenset()
        return wrap_if_filtering(compiled, suppressed)
