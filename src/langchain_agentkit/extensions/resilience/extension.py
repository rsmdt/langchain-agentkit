"""Write-time resilience for tool execution (Layer 1).

Rationale
---------
LangGraph's default tool-error handler re-raises non-``ToolException``
errors. When an exception propagates out of the tool node, LangGraph
aborts the turn **after** the model node has already committed its
``AIMessage(tool_calls=[...])`` to the checkpoint. Result: a permanent
orphan — an assistant tool call with no paired ``ToolMessage``.

On any subsequent turn, the OpenAI Responses API rejects the request
with ``"No tool output found for function call <call_id>"`` because it
enforces the pairing server-side.

This extension attaches a ``wrap_tool`` hook that catches any unhandled
exception and returns a synthetic ``ToolMessage`` with the same
``tool_call_id``. The pairing invariant is preserved at write time; no
orphan is ever checkpointed.

Scope and safety
----------------
- ``ToolException`` is re-raised: the tool's typed error contract is
  preserved for downstream consumers (tests, custom error handling).
- ``asyncio.CancelledError`` is re-raised: cancellation semantics
  (graceful shutdown, task-group teardown) are never swallowed.
- Only ``Exception`` subclasses are caught — ``BaseException`` escapes
  (``KeyboardInterrupt``, ``SystemExit``, ``GeneratorExit``).

Observability
-------------
Every catch emits a ``ToolErrorEvent`` via:
    1. Structured WARN log on ``langchain_agentkit.extensions.resilience``.
    2. Optional ``on_tool_error_caught`` callback for custom sinks
       (metrics, Sentry, PagerDuty).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.resilience.types import ToolErrorEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_logger = logging.getLogger(__name__)


def _default_error_template(exc: Exception, tool_name: str) -> str:
    """Default synthetic ToolMessage content.

    Deliberately short and deterministic so model behaviour is
    predictable across failures. The real exception details go to logs
    and telemetry, not to the LLM.
    """
    return f"[tool '{tool_name}' failed: {type(exc).__name__}]"


class ResilienceExtension(Extension):
    """Self-healing tool execution (Layer 1: write-time prevention).

    Args:
        tool_error_template: Callable ``(exc, tool_name) -> str`` that
            produces the synthetic ``ToolMessage`` content shown to the
            model. Defaults to a short, deterministic message.
        on_tool_error_caught: Optional callback invoked with a
            :class:`ToolErrorEvent` for every exception caught. Use for
            forwarding to metrics, Sentry, etc. Must not raise.
        include_exception_message: When True (default), appends the
            exception's ``str(exc)`` to the synthetic ToolMessage
            content. Turn off if exception messages may contain
            sensitive data.
    """

    def __init__(
        self,
        *,
        tool_error_template: Callable[[Exception, str], str] | None = None,
        on_tool_error_caught: Callable[[ToolErrorEvent], None] | None = None,
        include_exception_message: bool = True,
    ) -> None:
        self._template = tool_error_template or _default_error_template
        self._on_error = on_tool_error_caught
        self._include_exception_message = include_exception_message

    async def wrap_tool(
        self,
        *,
        state: Any,
        handler: Callable[[Any], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        """Catch unhandled tool exceptions and convert them to ToolMessages.

        Sits outermost in the tool-hook chain so it wraps every other
        extension's tool behaviour (HITL approvals, introspection,
        etc.). Whatever they do, if it raises, the pairing invariant
        still holds.
        """
        try:
            return await handler(state)
        except ToolException:
            # Tool's typed error contract — let ToolNode handle it.
            raise
        except asyncio.CancelledError:
            # Cancellation is a control signal, not a failure.
            raise
        except Exception as exc:
            return self._build_synthetic_message(exc, state)

    def _build_synthetic_message(self, exc: Exception, state: Any) -> ToolMessage:
        tool_call = _extract_tool_call(state)
        tool_call_id = tool_call.get("id", "")
        tool_name = tool_call.get("name", "")

        content = self._template(exc, tool_name)
        if self._include_exception_message:
            exc_str = str(exc).strip()
            if exc_str:
                content = f"{content} {exc_str}"

        event = ToolErrorEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            exc_type=type(exc).__name__,
            exc_message=str(exc),
            occurred_at=datetime.now(UTC),
        )
        self._emit_event(event, exc)

        return ToolMessage(
            content=content,
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
            additional_kwargs={
                "agentkit": {
                    "synthesized": True,
                    "reason": "tool_error",
                    "exc_type": event.exc_type,
                }
            },
        )

    def _emit_event(self, event: ToolErrorEvent, exc: Exception) -> None:
        _logger.warning(
            "resilience: tool '%s' raised %s; synthesized error ToolMessage (tool_call_id=%s)",
            event.tool_name,
            event.exc_type,
            event.tool_call_id,
            exc_info=exc,
            extra={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "exc_type": event.exc_type,
            },
        )
        if self._on_error is None:
            return
        try:
            self._on_error(event)
        except Exception:
            # Telemetry sinks must never break agent execution.
            _logger.exception("resilience: on_tool_error_caught callback raised")


def _extract_tool_call(state: Any) -> dict[str, Any]:
    """Pull the ``tool_call`` dict from a ToolCallRequest-shaped state.

    wrap_tool receives a ``ToolCallRequest`` (or dict) whose ``tool_call``
    attribute/key is a dict with ``id``, ``name``, ``args``. Defensive
    extraction so the extension stays robust to minor upstream changes
    and to the legacy dict-state shape.
    """
    tool_call = getattr(state, "tool_call", None)
    if tool_call is None and isinstance(state, dict):
        tool_call = state.get("tool_call")
    if tool_call is None:
        return {}
    if isinstance(tool_call, dict):
        return tool_call
    # Pydantic/TypedDict-like object
    return {
        "id": getattr(tool_call, "id", ""),
        "name": getattr(tool_call, "name", ""),
        "args": getattr(tool_call, "args", {}),
    }
