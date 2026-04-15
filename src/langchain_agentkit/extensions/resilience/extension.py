"""Self-healing tool execution and message invariants (Layers 1 + 2).

Rationale
---------
LangGraph's default tool-error handler re-raises non-``ToolException``
errors. When an exception propagates out of the tool node, LangGraph
aborts the turn **after** the model node has already committed its
``AIMessage(tool_calls=[...])`` to the checkpoint. Result: a permanent
orphan — an assistant tool call with no paired ``ToolMessage``.

On any subsequent turn, the OpenAI Responses API rejects the request
with ``"No tool output found for function call <call_id>"`` because it
enforces the pairing server-side. Orphans also arise from causes this
extension cannot prevent: pod kills between checkpoint writes,
checkpointer transient failures, manual state edits, replays of
half-finished turns.

Two hooks, one extension:

* **Layer 1 — ``wrap_tool``**: catches any unhandled tool exception
  and synthesizes a paired ``ToolMessage`` on the fly. Prevents new
  orphans from ever being written.
* **Layer 2 — ``wrap_model``**: before the LLM call, scans the message
  list for ``AIMessage(tool_calls=[...])`` whose ``tool_call_id``s have
  no matching ``ToolMessage`` and injects synthetic ones. Repairs
  orphans that pre-existed (from earlier crashes, infrastructure
  failures, or edits) so the model request is always well-formed.

Layer 2 fires only from ``wrap_model`` — i.e. when the graph is
entering the model node. Pending HITL interrupts pause the graph
before the model node, so legitimate in-flight tool calls are never
misidentified as orphans.

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

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import ToolException

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.resilience.types import (
    OrphanRepairEvent,
    ToolErrorEvent,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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
        repair_orphan_tool_calls: bool = True,
        orphan_repair_message: str = (
            "[tool result unavailable — prior turn aborted before a response was recorded]"
        ),
        on_orphan_repaired: Callable[[OrphanRepairEvent], None] | None = None,
    ) -> None:
        self._template = tool_error_template or _default_error_template
        self._on_error = on_tool_error_caught
        self._include_exception_message = include_exception_message
        self._repair_orphans = repair_orphan_tool_calls
        self._orphan_message = orphan_repair_message
        self._on_orphan = on_orphan_repaired

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

    async def wrap_model(
        self,
        *,
        state: Any,
        handler: Callable[[Any], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        """Repair orphan tool calls before the LLM sees the message list.

        An orphan is an ``AIMessage(tool_calls=[...])`` whose
        ``tool_call_id`` has no paired ``ToolMessage``. They originate
        from crashed tool executions that pre-date the Layer 1 hook, pod
        kills between checkpoint writes, or manual state edits. The
        OpenAI Responses API rejects orphans with ``"No tool output
        found for function call"``.

        The repair inserts synthetic ``ToolMessage`` entries immediately
        after each offending ``AIMessage``. Downstream extensions
        (``HistoryExtension``, ``ContextCompactionExtension``) see the
        repaired list and treat it as a complete turn.

        This hook only fires from ``wrap_model`` — i.e. when the graph
        is entering the model node. Pending HITL interrupts pause the
        graph before the model node, so legitimate in-flight tool calls
        are never misidentified as orphans.
        """
        if not self._repair_orphans:
            return await handler(state)

        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return await handler(state)

        repaired, repairs = self._repair_orphan_messages(messages)
        if not repairs:
            return await handler(state)

        for repair in repairs:
            self._emit_repair(repair)

        return await handler({**state, "messages": repaired})

    def _repair_orphan_messages(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[list[BaseMessage], list[OrphanRepairEvent]]:
        """Insert synthetic ToolMessages for unpaired AIMessage tool_calls.

        Walks the list once. Maintains a set of ``tool_call_id``s already
        satisfied by a ``ToolMessage`` seen *before* the current position.
        When an ``AIMessage`` with ``tool_calls`` is encountered, any
        ``tool_call_id`` not yet satisfied AND not appearing in a
        ``ToolMessage`` later in the list is synthesized on the spot so
        the pairing is contiguous (required by OpenAI Responses API and
        by block-aware history strategies).
        """
        future_outputs = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        out: list[BaseMessage] = []
        repairs: list[OrphanRepairEvent] = []
        satisfied: set[str] = set()

        for msg in messages:
            if isinstance(msg, ToolMessage):
                satisfied.add(msg.tool_call_id)

            out.append(msg)

            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                continue

            synthesized_here: list[ToolMessage] = []
            for tc in msg.tool_calls:
                call_id = tc.get("id") or ""
                if not call_id or call_id in satisfied:
                    continue
                # If a matching ToolMessage exists *later* in the list,
                # it will satisfy the pairing when we reach it. No repair
                # needed now.
                if call_id in future_outputs:
                    continue
                tool_name = tc.get("name") or ""
                synthetic = ToolMessage(
                    content=self._orphan_message,
                    name=tool_name,
                    tool_call_id=call_id,
                    status="error",
                    additional_kwargs={
                        "agentkit": {
                            "synthesized": True,
                            "reason": "orphan",
                        }
                    },
                )
                synthesized_here.append(synthetic)
                satisfied.add(call_id)
                repairs.append(
                    OrphanRepairEvent(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        ai_message_id=msg.id,
                        repaired_at=datetime.now(UTC),
                    )
                )
            out.extend(synthesized_here)

        return out, repairs

    def _emit_repair(self, event: OrphanRepairEvent) -> None:
        _logger.warning(
            "resilience: repaired orphan tool call '%s' (tool=%s, ai_message_id=%s); "
            "synthesized ToolMessage for the model request",
            event.tool_call_id,
            event.tool_name,
            event.ai_message_id,
            extra={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "ai_message_id": event.ai_message_id,
                "reason": "orphan",
            },
        )
        if self._on_orphan is None:
            return
        try:
            self._on_orphan(event)
        except Exception:
            _logger.exception("resilience: on_orphan_repaired callback raised")

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
