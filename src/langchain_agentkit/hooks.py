"""Hook decorators for Extension lifecycle interception.

Three decorators mark methods as hooks:

    @before("model")          — runs before the hook point
    @after("tool")            — runs after the hook point
    @wrap("model")            — onion-style wrapper around the hook point

All support per-tool filtering via the ``tools`` parameter::

    @wrap("tool", tools=["delete_file", "send_email"])
    async def gate(self, request, handler):
        ...
"""

from __future__ import annotations

from typing import Callable


def before(point: str, *, tools: list[str] | None = None) -> Callable:
    """Mark a method as a before-hook for the given point.

    Args:
        point: Hook point — "run", "model", or "tool".
        tools: Optional list of tool names to filter on (tool hooks only).
    """

    def decorator(fn: Callable) -> Callable:
        fn._hook_phase = "before"  # type: ignore[attr-defined]
        fn._hook_point = point  # type: ignore[attr-defined]
        fn._hook_tool_filter = tools  # type: ignore[attr-defined]
        return fn

    return decorator


def after(point: str, *, tools: list[str] | None = None) -> Callable:
    """Mark a method as an after-hook for the given point.

    Args:
        point: Hook point — "run", "model", or "tool".
        tools: Optional list of tool names to filter on (tool hooks only).
    """

    def decorator(fn: Callable) -> Callable:
        fn._hook_phase = "after"  # type: ignore[attr-defined]
        fn._hook_point = point  # type: ignore[attr-defined]
        fn._hook_tool_filter = tools  # type: ignore[attr-defined]
        return fn

    return decorator


def wrap(point: str, *, tools: list[str] | None = None) -> Callable:
    """Mark a method as an onion-style wrapper for the given point.

    Args:
        point: Hook point — "run", "model", or "tool".
        tools: Optional list of tool names to filter on (tool hooks only).
    """

    def decorator(fn: Callable) -> Callable:
        fn._hook_phase = "wrap"  # type: ignore[attr-defined]
        fn._hook_point = point  # type: ignore[attr-defined]
        fn._hook_tool_filter = tools  # type: ignore[attr-defined]
        return fn

    return decorator
