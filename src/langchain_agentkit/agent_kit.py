"""AgentKit — sync composition engine for Extension instances.

``AgentKit`` is a self-contained composition engine with a clean public API.
It merges tools, prompts, model, and hooks from extensions into a unified
surface. ``compile(handler)`` builds a complete ReAct graph with hooks wired.

**Managed graph** — ``compile(handler)`` builds the full ReAct loop::

    kit = AgentKit(
        extensions=[SkillsExtension(skills="skills/"), TasksExtension()],
        tools=[web_search],
        model=ChatOpenAI(model="gpt-4o"),
        prompt="You are a research assistant.",
    )
    graph = kit.compile(handler)      # uncompiled StateGraph
    app = graph.compile()             # compiled, ready to invoke

**Manual wiring** — access components directly for custom graphs::

    kit = AgentKit(extensions=exts)
    kit.tools          # merged user + extension tools
    kit.prompt(state)  # composed system prompt
    kit.model          # resolved BaseChatModel
    kit.state_schema   # composed TypedDict from extensions
    kit.hooks          # HookRunner for manual lifecycle wiring
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.extension import Extension

_logger = logging.getLogger(__name__)


class AgentKit:
    """Sync composition engine — merges extensions, tools, model, and prompt.

    Use ``compile(handler)`` to build a managed ReAct graph, or access
    ``tools``, ``prompt()``, ``model``, ``state_schema``, and ``hooks``
    directly for manual graph wiring.

    Args:
        extensions: Ordered list of Extension instances.
        prompt: System prompt — inline string, file path, or list.
        tools: User-provided tools (merged with extension tools).
        model: ``BaseChatModel`` instance or string (resolved via
            ``model_resolver``).
        model_resolver: Callable resolving model name strings to
            ``BaseChatModel`` instances.
        name: Graph node name (default ``"agent"``).

    Example::

        kit = AgentKit(
            extensions=[SkillsExtension(skills="skills/"), TasksExtension()],
            model=ChatOpenAI(model="gpt-4o"),
            prompt="You are a research assistant.",
        )
        graph = kit.compile(handler)
    """

    def __init__(
        self,
        *,
        extensions: list[Extension] | None = None,
        prompt: str | Path | list[str | Path] | None = None,
        tools: list[BaseTool] | None = None,
        model: BaseChatModel | str | None = None,
        model_resolver: Callable[[str], BaseChatModel] | None = None,
        name: str = "agent",
    ) -> None:
        self._extensions = self._resolve_dependencies(list(extensions or []))
        self._prompt = _load_prompt(prompt)
        self._user_tools: list[BaseTool] = list(tools or [])
        self._model_raw = model
        self._model_resolver = model_resolver
        self._name = name
        self._tools_cache: list[BaseTool] | None = None

    @staticmethod
    def _resolve_dependencies(extensions: list[Any]) -> list[Any]:
        """Resolve extension dependencies. Auto-add missing dependencies.

        Uses isinstance/type for identity. Dependencies declared via
        the optional dependencies() method are added if not already present.
        """
        resolved = list(extensions)
        seen_types = {type(ext) for ext in resolved}

        # Iterate a copy — resolved may grow during iteration
        for ext in list(resolved):
            deps_fn = getattr(ext, "dependencies", None)
            if not callable(deps_fn):
                continue
            for dep in deps_fn():
                if type(dep) not in seen_types:
                    resolved.append(dep)
                    seen_types.add(type(dep))

        return resolved

    @property
    def tools(self) -> list[BaseTool]:
        """Merged user + extension tools, deduplicated by name.

        User tools appear first, then extension tools in stack order.
        First tool wins on name collisions.
        Collected once on first access, then cached.
        """
        if self._tools_cache is None:
            seen: set[str] = set()
            tools: list[BaseTool] = []
            for tool in self._user_tools:
                if tool.name not in seen:
                    seen.add(tool.name)
                    tools.append(tool)
            for ext in self._extensions:
                for tool in ext.tools:
                    if tool.name not in seen:
                        seen.add(tool.name)
                        tools.append(tool)
            self._tools_cache = tools
        return self._tools_cache

    @property
    def state_schema(self) -> type:
        """Compose state schema from ``AgentKitState`` + all extension schemas.

        Each extension may declare a ``state_schema`` property returning a
        TypedDict mixin. These are combined via multiple inheritance into a
        single composed type. Extensions without ``state_schema`` (or returning
        ``None``) are skipped.
        """
        from langchain_agentkit.state import AgentKitState

        bases: list[type] = [AgentKitState]
        seen: set[int] = {id(AgentKitState)}
        for ext in self._extensions:
            schema = getattr(ext, "state_schema", None)
            if schema is not None and id(schema) not in seen:
                seen.add(id(schema))
                bases.append(schema)
        if len(bases) == 1:
            return AgentKitState
        return type("ComposedState", tuple(bases), {})

    @property
    def hooks(self) -> Any:
        """Collected HookRunner for manual hook wiring.

        Returns a ``HookRunner`` built from all extensions. Use for
        manual graph construction when ``compile(handler)`` is too
        opinionated.
        """
        from langchain_agentkit.hook_runner import HookRunner

        return HookRunner(self._extensions)

    @property
    def model(self) -> Any:
        """Resolved BaseChatModel instance.

        Applies the fallback chain:
        1. ``model_resolver(name)`` if model is a string and resolver exists
        2. Falls back to ``self._model_raw`` as-is for non-string models
        3. Raises if model is a string and no resolver is available

        Raises:
            ValueError: If model is a string but no resolver is configured
                and no extension provides one.
        """
        return self._resolve_model_internal(self._model_raw)

    def _resolve_model_internal(self, model: Any) -> Any:
        """Apply the model resolver fallback chain.

        1. model_resolver(name) succeeds → use the result
        2. model_resolver(name) raises or returns None → warn, fall back to self._model_raw
        3. No model_resolver → scan extensions, fall back to self._model_raw
        4. No model at all → error
        """
        if not isinstance(model, str):
            return model

        # Try kit-level resolver first
        if self._model_resolver is not None:
            try:
                result = self._model_resolver(model)
                if result is not None:
                    return result
            except Exception:
                _logger.warning(
                    "model_resolver raised for '%s', no fallback model available",
                    model,
                )
                raise

        # Fallback: scan extensions for a model_resolver attribute
        for ext in self._extensions:
            resolver = getattr(ext, "model_resolver", None)
            if callable(resolver):
                return resolver(model)

        raise ValueError(
            f"model='{model}' is a string but no model_resolver is configured "
            f"and no extension provides one. Pass model_resolver=<callable> to "
            f"AgentKit or use a BaseChatModel instance."
        )

    def resolve_model(self, model: Any) -> Any:
        """Resolve a model reference to a BaseChatModel instance.

        If *model* is a string, uses the configured ``model_resolver`` or
        scans extensions for one that exposes a ``model_resolver`` attribute.
        Otherwise returns the input as-is (assumed to be a BaseChatModel).

        Raises:
            ValueError: If *model* is a string but no resolver is available.
        """
        return self._resolve_model_internal(model)

    def compile(self, handler: Any) -> Any:
        """Build the full ReAct graph with hooks wired.

        Absorbs all logic from ``build_graph()``. Returns an uncompiled
        ``StateGraph``. Call ``.compile()`` on the result to get a runnable
        graph, optionally passing a checkpointer for ``interrupt()`` support.

        Args:
            handler: The agent handler function. Accepts ``state`` as first
                positional arg, plus keyword-only injectables (llm, tools,
                prompt, runtime).

        Returns:
            An uncompiled ``StateGraph``.
        """
        from langchain_agentkit._graph_builder import _find_wrap_tool_call, build_graph

        llm = self._resolve_model_internal(self._model_raw) if self._model_raw is not None else None
        wrap_tool_call = _find_wrap_tool_call(self._extensions, self._name)

        # Pass user_tools=[] because self.tools already merges user + extension tools.
        # build_graph does `all_tools = user_tools + kit.tools`, so passing []
        # avoids duplicating the user tools that are already in self.tools.
        return build_graph(
            name=self._name,
            handler=handler,
            llm=llm,  # type: ignore[arg-type]
            user_tools=[],
            kit=self,
            state_type=self.state_schema,
            wrap_tool_call=wrap_tool_call,
        )

    def _collect_contributions(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> tuple[list[str], list[str]]:
        """Single-pass collection of prompt sections and reminders.

        Iterates each extension's ``prompt()`` exactly once, splitting
        the result into system-prompt sections and ephemeral reminders.

        Returns:
            (prompt_parts, reminder_parts) — both are lists of non-empty strings.

        Extension ``prompt()`` can return:
        - ``str`` — goes into system prompt
        - ``dict`` with ``prompt`` and/or ``reminder`` keys — prompt goes
          to system prompt, reminder goes to ephemeral message
        - ``None`` — no contribution
        """
        prompt_parts: list[str] = [self._prompt] if self._prompt else []
        reminder_parts: list[str] = []
        for ext in self._extensions:
            result = ext.prompt(state, runtime)
            if result is None:
                continue
            if isinstance(result, str):
                if result:
                    prompt_parts.append(result)
            elif isinstance(result, dict):
                p = result.get("prompt", "")
                r = result.get("reminder", "")
                if p:
                    prompt_parts.append(p)
                if r:
                    reminder_parts.append(r)
        return prompt_parts, reminder_parts

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Compose system prompt from template + extension prompt sections.

        Called on every LLM invocation. Each extension contributes
        a section in stack order. Sections joined with double newline.

        Extensions returning a dict with ``prompt``/``reminder`` keys
        have only their ``prompt`` value collected here. The ``reminder``
        value is collected separately by :meth:`system_reminder`.
        """
        prompt_parts, _ = self._collect_contributions(state, runtime)
        return "\n\n".join(prompt_parts)

    def system_reminder(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Collect ephemeral reminder content from all extensions.

        Returns combined text that will be appended to the messages
        array as a ``HumanMessage`` wrapped in ``<system-reminder>``
        tags. This message is added at LLM call time and **never**
        stored in ``state["messages"]``.

        Only extensions returning a dict with a non-empty
        ``reminder`` value contribute here.
        """
        _, reminder_parts = self._collect_contributions(state, runtime)
        return "\n\n".join(reminder_parts) if reminder_parts else ""


async def run_extension_setup(kit: AgentKit) -> None:
    """Run setup() on all extensions in a kit.

    Call this before using the kit when extensions have async setup
    (backend discovery, cross-extension wiring). This is the only
    async operation in the AgentKit lifecycle — it lives outside
    the class to keep AgentKit purely sync.

    Supports both sync and async setup methods. Each extension declares
    only the kwargs it needs via its own ``setup()`` signature.
    """
    available: dict[str, Any] = {
        "extensions": kit._extensions,
        "prompt": kit._prompt,
        "model_resolver": kit._model_resolver,
    }
    for ext in kit._extensions:
        setup = getattr(ext, "setup", None)
        if not callable(setup):
            continue
        try:
            sig_params = inspect.signature(setup).parameters
        except (TypeError, ValueError):
            continue
        accepts_var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params.values()
        )
        if accepts_var_keyword:
            kwargs = dict(available)
        else:
            kwargs = {k: v for k, v in available.items() if k in sig_params}
        result = setup(**kwargs)
        if inspect.isawaitable(result):
            await result
    # Extensions may have rebuilt their tools during setup — invalidate.
    kit._tools_cache = None


def _load_prompt_source(source: str | Path) -> str:
    """Load a single prompt source from file path or return inline string."""
    text = str(source)
    # Skip path resolution for strings that are clearly inline prompts
    # (contain newlines or exceed OS filename limits).
    if "\n" in text or len(text) > 255:
        return text
    path = Path(text)
    try:
        if path.exists() and path.is_file():
            return path.read_text()
    except OSError:
        pass
    return text


def _load_prompt(source: str | Path | list[str | Path] | None) -> str:
    """Load and concatenate prompt sources."""
    if source is None:
        return ""
    if isinstance(source, (str, Path)):
        return _load_prompt_source(source)
    parts = [_load_prompt_source(s) for s in source]
    return "\n\n".join(p for p in parts if p)
