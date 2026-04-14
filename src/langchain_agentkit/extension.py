"""Extension base class for composable agent capabilities.

An Extension bundles tools, prompts, state schema, lifecycle hooks,
and graph modifications into a cohesive, reusable package.

Simple case — override named methods::

    class SecurityExtension(Extension):
        async def before_model(self, *, state, runtime):
            return sanitize_messages(state)

Advanced case — decorators with per-tool filtering::

    class GovernanceExtension(Extension):
        @wrap("tool", tools=["delete_file"])
        async def require_approval(self, *, state, handler, runtime):
            return interrupt(state)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Literal

# Named method patterns recognized as hooks
_NAMED_HOOK_METHODS: dict[str, tuple[str, str]] = {
    "before_run": ("before", "run"),
    "after_run": ("after", "run"),
    "on_error": ("on_error", "run"),
    "before_model": ("before", "model"),
    "after_model": ("after", "model"),
    "wrap_model": ("wrap", "model"),
    "before_tool": ("before", "tool"),
    "after_tool": ("after", "tool"),
    "wrap_tool": ("wrap", "tool"),
}


class Extension:
    """Base class for composable agent capabilities.

    An Extension contributes tools, prompts, state schema, lifecycle hooks,
    and graph modifications to an agent. Subclass and override what you need.

    Hook registration:
        - **Named methods**: Override ``before_model``, ``wrap_tool``, etc.
        - **Decorators**: Use ``@before("model")``, ``@wrap("tool", tools=[...])``, etc.
          for per-tool filtering and multiple hooks on the same class.

    Both styles can coexist on the same class. Decorated hooks are collected
    via ``__init_subclass__`` at class definition time.
    """

    _decorated_hooks: dict[tuple[str, str], list[Any]]

    #: Cache scope for this extension's default ``prompt()`` contribution.
    #:
    #: When ``prompt()`` returns a plain string (not a ``dict``), the string is
    #: routed to either the ``static`` or ``dynamic`` section of the composed
    #: system prompt according to this attribute. Extensions that emit stable,
    #: rarely-changing content should set this to ``"static"`` to maximize
    #: prompt-cache reuse; extensions whose prompts render live state should
    #: leave the default of ``"dynamic"``.
    prompt_cache_scope: ClassVar[Literal["static", "dynamic"]] = "dynamic"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hooks: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for attr_name in list(vars(cls)):
            method = vars(cls).get(attr_name)
            if callable(method) and hasattr(method, "_hook_phase"):
                key = (method._hook_phase, method._hook_point)
                hooks[key].append(method)
        cls._decorated_hooks = dict(hooks)

    # --- Required protocol (with defaults) ---

    @property
    def tools(self) -> list[Any]:
        """Tools this extension provides to the agent."""
        return []

    def prompt(
        self, state: dict[str, Any], runtime: Any | None = None
    ) -> str | dict[str, str] | None:
        """Prompt section contributed by this extension.

        Called on every LLM invocation. May return:

        - ``None`` or ``""`` — contribute nothing.
        - ``str`` — contribute a single section, routed to the scope declared
          by :attr:`prompt_cache_scope` (``"static"`` or ``"dynamic"``).
        - ``dict`` with any of the keys ``"prompt"`` and ``"reminder"``:

          * ``"prompt"`` is routed by :attr:`prompt_cache_scope`.
          * ``"reminder"`` is appended to AgentKit's ``<system-reminder>``
            envelope under a ``# <ExtensionClassName>`` header.

          Unknown keys are silently ignored.
        """
        return None

    @property
    def state_schema(self) -> type | None:
        """TypedDict mixin for this extension's state requirements.

        Return a TypedDict class to add keys to the graph state, or
        None if no additional state keys are needed.
        """
        return None

    def dependencies(self) -> list[Any]:
        """Optional: extensions this one depends on. Auto-added if missing.

        Return instances of required extensions. The dependency resolver
        deduplicates by type.
        """
        return []

    def setup(self, **kwargs: Any) -> None:
        """Finalize configuration once the kit is assembled.

        Called once by :class:`AgentKit` after dependency resolution and
        before the graph is built.  Override to react to other extensions
        or capture kit-level config.

        The framework calls this method with keyword arguments drawn from
        the ``AgentKit`` configuration.  Each extension declares only the
        parameters it needs via its own ``setup()`` signature — the
        framework inspects the signature and passes only the matching
        kwargs.  Currently available::

            def setup(self, *, extensions: list[Extension]) -> None: ...
            def setup(self, *, extensions, prompt) -> None: ...

        ``extensions`` is the full ordered list of extensions in the kit,
        including ``self``.  ``prompt`` is the base prompt configured on
        ``AgentKit`` (an empty string if none was provided).

        **Contract — inspect presence, not state:**

        ``setup()`` is called in declaration order, which means another
        extension's ``setup()`` may not have run yet when yours executes.
        You MUST only inspect the *presence* of sibling extensions (via
        ``isinstance()`` checks) — never read mutable state that another
        extension's ``setup()`` might populate.  Anything that depends on
        another extension being fully configured should happen lazily,
        at runtime, not during ``setup()``.

        **Error handling:** Exceptions raised from ``setup()`` propagate
        out of ``AgentKit.__init__``.  Setup failures are intentionally
        fatal — a mis-configured extension should block kit construction
        rather than silently degrade at runtime.

        **Idempotency:** ``setup()`` is called exactly once per kit.  If
        the same instance is added to multiple kits, it will be called
        once per kit — keep overrides idempotent (safe to call more than
        once with the same or different sibling sets).
        """

    # --- Hook discovery ---

    @classmethod
    def _get_named_hooks(cls) -> dict[tuple[str, str], Any]:
        """Discover named hook methods on this class.

        Returns a dict mapping (phase, point) to the unbound method,
        only for methods actually overridden on this class (not inherited
        from Extension base).
        """
        result: dict[tuple[str, str], Any] = {}
        for method_name, key in _NAMED_HOOK_METHODS.items():
            method = getattr(cls, method_name, None)
            if method is None:
                continue
            # Only include if overridden (not the base Extension class)
            if method_name not in vars(cls):
                continue
            result[key] = method
        return result

    def get_all_hooks(self) -> dict[tuple[str, str], list[Any]]:
        """Get all hooks (named methods + decorated) as bound methods.

        Returns a dict mapping (phase, point) to a list of bound methods.
        Named methods come first, then decorated hooks.
        """
        result: dict[tuple[str, str], list[Any]] = defaultdict(list)

        # Named methods first
        for key, unbound in self.__class__._get_named_hooks().items():
            result[key].append(getattr(self, unbound.__name__))

        # Then decorated hooks
        for key, unbound_list in self.__class__._decorated_hooks.items():
            for unbound in unbound_list:
                result[key].append(getattr(self, unbound.__name__))

        return dict(result)
