"""HookRunner — collects and executes hooks from Extension instances.

Handles the three composition patterns:
- ``before_*``: runs in declaration order, collects state updates
- ``after_*``: runs in reverse declaration order
- ``wrap_*``: composes as onion layers (first extension = outermost)

Also handles:
- ``on_error``: fires on unhandled errors
- Per-tool filtering for all tool hooks (before, after, wrap)
- ``jump_to`` routing from before_model/after_model hooks
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_VALID_JUMP_TARGETS = frozenset({"end", "model", "tools"})

# Type alias to keep signatures under 100 chars
_HookEntry = tuple[Any, "Callable[..., Any]", list[str] | None]


class HookRunner:
    """Collects hooks from extensions and provides execution methods.

    Args:
        extensions: Ordered list of Extension instances.
    """

    def __init__(self, extensions: list[Any]) -> None:
        self._extensions = list(extensions)
        self._hooks = self._collect_hooks()
        self._error_hooks = self._collect_error_hooks()

    def _collect_hooks(self) -> dict[tuple[str, str], list[_HookEntry]]:
        """Collect all hooks from all extensions.

        Returns a dict mapping (phase, point) to a list of
        (extension_instance, bound_method, tool_filter) tuples.
        """
        hooks: dict[tuple[str, str], list[_HookEntry]] = defaultdict(list)

        for ext in self._extensions:
            get_hooks = getattr(ext, "get_all_hooks", None)
            if not callable(get_hooks):
                continue  # Extension doesn't support hooks (legacy or stub)
            all_hooks = get_hooks()
            for (phase, point), methods in all_hooks.items():
                if phase == "on_error":
                    continue  # Handled separately
                for method in methods:
                    tool_filter = getattr(method, "_hook_tool_filter", None)
                    # For bound methods wrapping unbound with filter, check the underlying function
                    if tool_filter is None and hasattr(method, "__func__"):
                        tool_filter = getattr(method.__func__, "_hook_tool_filter", None)
                    hooks[(phase, point)].append((ext, method, tool_filter))

        return dict(hooks)

    def _collect_error_hooks(self) -> list[Callable[..., Any]]:
        """Collect on_error hooks from extensions."""
        hooks = []
        for ext in self._extensions:
            get_hooks = getattr(ext, "get_all_hooks", None)
            if not callable(get_hooks):
                continue
            all_hooks = get_hooks()
            for method in all_hooks.get(("on_error", "run"), []):
                hooks.append(method)
        return hooks

    @property
    def has_run_hooks(self) -> bool:
        """True if any before_run, after_run, or on_error hooks are registered."""
        return bool(
            self._hooks.get(("before", "run"))
            or self._hooks.get(("after", "run"))
            or self._error_hooks
        )

    @staticmethod
    def _validate_jump_to(result: dict[str, Any]) -> None:
        """Validate jump_to target if present in a hook result."""
        if "jump_to" in result:
            target = result["jump_to"]
            if target not in _VALID_JUMP_TARGETS:
                raise ValueError(
                    f"Invalid jump_to target '{target}'. "
                    f"Valid targets: {', '.join(sorted(_VALID_JUMP_TARGETS))}"
                )

    @staticmethod
    def _matches_tool_filter(
        tool_filter: list[str] | None,
        tool_name: str | None,
    ) -> bool:
        """Check if a hook should fire given its tool filter and the current tool name."""
        if tool_filter is None:
            return True  # No filter — matches all
        if tool_name is None:
            return False  # Filter specified but no tool name — skip
        return tool_name in tool_filter

    async def run_before(
        self,
        point: str,
        *,
        state: dict[str, Any],
        runtime: Any,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run all before hooks for the given point in declaration order.

        Returns a list of state update dicts (excludes None returns).
        Supports per-tool filtering via ``tool_name`` for tool hooks.

        If a before hook returns ``{"jump_to": destination}``, the jump_to
        value is included in the returned update dict for the caller to handle.
        Valid destinations: "model", "tools", "end".
        """
        updates: list[dict[str, Any]] = []
        for _ext, method, tool_filter in self._hooks.get(("before", point), []):
            if not self._matches_tool_filter(tool_filter, tool_name):
                continue
            result = await method(state=state, runtime=runtime)
            if result is not None:
                self._validate_jump_to(result)
                updates.append(result)
        return updates

    async def run_after(
        self,
        point: str,
        *,
        state: dict[str, Any],
        runtime: Any,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run all after hooks for the given point in REVERSE declaration order.

        Returns a list of state update dicts (excludes None returns).
        Supports per-tool filtering via ``tool_name`` for tool hooks.

        If an after hook returns ``{"jump_to": destination}``, the jump_to
        value is included in the returned update dict for the caller to handle.
        Valid destinations: "model", "tools", "end".
        """
        updates: list[dict[str, Any]] = []
        hooks = list(self._hooks.get(("after", point), []))
        for _ext, method, tool_filter in reversed(hooks):
            if not self._matches_tool_filter(tool_filter, tool_name):
                continue
            result = await method(state=state, runtime=runtime)
            if result is not None:
                self._validate_jump_to(result)
                updates.append(result)
        return updates

    async def run_wrap(
        self,
        point: str,
        *,
        state: Any,
        handler: Callable[..., Any],
        runtime: Any,
        tool_name: str | None = None,
    ) -> Any:
        """Run wrap hooks as onion layers around the handler.

        Each hook receives ``state``, ``handler``, and ``runtime`` as
        keyword arguments — consistent with before/after hooks.

        First extension = outermost layer. Per-tool filtering applied
        when ``tool_name`` is provided.
        """
        wrap_hooks = self._hooks.get(("wrap", point), [])

        # Filter by tool name if applicable
        applicable = []
        for _ext, method, tool_filter in wrap_hooks:
            if not self._matches_tool_filter(tool_filter, tool_name):
                continue
            applicable.append(method)

        if not applicable:
            return await handler(state)

        # Build onion: last hook wraps the handler, first hook is outermost
        current_handler = handler
        for method in reversed(applicable):
            outer_handler = current_handler

            async def make_layer(
                m: Callable[..., Any],
                h: Callable[..., Any],
                rt: Any,
            ) -> Callable[..., Any]:
                async def layer(st: Any) -> Any:
                    return await m(state=st, handler=h, runtime=rt)

                return layer

            current_handler = await make_layer(method, outer_handler, runtime)

        return await current_handler(state)

    async def run_on_error(
        self,
        error: Exception,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        """Run on_error hooks."""
        for hook in self._error_hooks:
            await hook(error=error, state=state, runtime=runtime)
