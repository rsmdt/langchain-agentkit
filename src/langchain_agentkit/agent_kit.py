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
    kit.tools                     # merged user + extension tools
    kit.compose(state, runtime)   # PromptComposition(prompt, reminder)
    kit.model                     # resolved BaseChatModel
    kit.state_schema   # composed TypedDict from extensions
    kit.hooks          # HookRunner for manual lifecycle wiring
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.prompt_composition import PromptComposition
from langchain_agentkit.reminders import assemble_builtin_reminder

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.extension import Extension

_logger = logging.getLogger(__name__)

_VALID_PRESETS: frozenset[str] = frozenset({"full"})


class AgentKit:
    """Sync composition engine — merges extensions, tools, model, and prompt.

    Use ``compile(handler)`` to build a managed ReAct graph, or access
    ``tools``, ``compose()``, ``model``, ``state_schema``, and ``hooks``
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
        preset: Optional preset that seeds a curated extension stack.
            When ``"full"``, the kit prepends
            ``[CoreBehaviorExtension(), TasksExtension()]`` ahead of
            any user-supplied ``extensions=[...]``. ``None`` (default)
            leaves the extension list untouched. Unknown preset
            strings raise :class:`ValueError`.

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
        preset: str | None = None,
    ) -> None:
        seeded = _seed_preset_extensions(preset)
        combined: list[Extension] = [*seeded, *(extensions or [])]
        self._extensions = self._resolve_dependencies(combined)
        self._prompt = _load_prompt(prompt)
        self._user_tools: list[BaseTool] = list(tools or [])
        self._model_raw = model
        self._model_resolver = model_resolver
        self._name = name
        self._tools_cache: list[BaseTool] | None = None

    @property
    def extensions(self) -> list[Extension]:
        """Ordered list of extensions (after dependency resolution)."""
        return list(self._extensions)

    @property
    def base_prompt(self) -> str:
        """The resolved base prompt string (before extension contributions)."""
        return self._prompt

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
        """
        return self._resolve_model_internal(self._model_raw)

    def _resolve_model_internal(self, model: Any) -> Any:
        """Apply the model resolver fallback chain."""
        if not isinstance(model, str):
            return model

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
        """Resolve a model reference to a BaseChatModel instance."""
        return self._resolve_model_internal(model)

    def compile(self, handler: Any) -> Any:
        """Build the full ReAct graph with hooks wired."""
        from langchain_agentkit._graph_builder import build_graph

        llm = self._resolve_model_internal(self._model_raw) if self._model_raw is not None else None

        return build_graph(
            name=self._name,
            handler=handler,
            llm=llm,  # type: ignore[arg-type]
            user_tools=[],
            kit=self,
            state_type=self.state_schema,
        )

    def compose(
        self, state: dict[str, Any], runtime: ToolRuntime | None = None
    ) -> PromptComposition:
        """Compose the per-step system prompt and reminder envelope.

        Iterates each extension's ``prompt()`` exactly once:

        - The kit-level base prompt is emitted first.
        - ``str`` returns are appended to the prompt channel.
        - ``dict`` returns may carry ``"prompt"`` and ``"reminder"`` keys.
          ``"prompt"`` is appended to the prompt channel; ``"reminder"``
          is appended to the reminder channel as a
          ``# <ext-class-name>`` section. Unknown keys are ignored.
        - ``None`` and empty-string returns contribute nothing.

        The reminder channel always starts with AgentKit's built-in
        source (today's date) followed by any extension contributions.
        When every source is empty, the ``reminder`` field is the
        empty string.

        Prompt sections are joined with ``"\\n\\n"`` in extension
        declaration order.
        """
        prompt_parts: list[str] = [self._prompt] if self._prompt else []
        reminder_sections: list[str] = []

        for ext in self._extensions:
            result = ext.prompt(state, runtime)
            if result is None:
                continue
            if isinstance(result, str):
                if result:
                    prompt_parts.append(result)
            elif isinstance(result, dict):
                prompt_piece = result.get("prompt")
                if isinstance(prompt_piece, str) and prompt_piece:
                    prompt_parts.append(prompt_piece)
                reminder_piece = result.get("reminder")
                if isinstance(reminder_piece, str) and reminder_piece:
                    key = type(ext).__name__
                    reminder_sections.append(f"# {key}\n{reminder_piece}")

        reminder = assemble_builtin_reminder(reminder_sections)
        return PromptComposition(
            prompt="\n\n".join(prompt_parts),
            reminder=reminder,
        )

    def system_reminder(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Return the reminder channel from :meth:`compose`."""
        return self.compose(state, runtime).reminder


def _seed_preset_extensions(preset: str | None) -> list[Any]:
    """Return the preset-seeded extension list (empty when no preset)."""
    if preset is None:
        return []
    if preset not in _VALID_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Valid presets: {sorted(_VALID_PRESETS)!r}.")
    if preset == "full":
        from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension
        from langchain_agentkit.extensions.tasks import TasksExtension

        return [
            CoreBehaviorExtension(),
            TasksExtension(),
        ]
    return []  # pragma: no cover — guarded by _VALID_PRESETS


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
        # Lazy getters so extensions that need the parent LLM or merged tool
        # set (e.g. AgentsExtension, TeamExtension) don't have to import or
        # know about AgentKit. Evaluated at tool-call time so they see the
        # fully-resolved model and the final merged tool list.
        "llm_getter": lambda: kit.model,
        "tools_getter": lambda: kit.tools,
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
