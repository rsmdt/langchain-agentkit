"""Typed events emitted by :class:`ResilienceExtension`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class ToolErrorEvent:
    """Emitted when the resilience layer catches a tool-execution exception.

    The resilience layer converts the exception into a synthetic
    ``ToolMessage`` so the AIMessage↔ToolMessage pairing invariant stays
    intact. This event lets operators see the *real* failure behind that
    synthetic message.
    """

    tool_call_id: str
    tool_name: str
    exc_type: str
    exc_message: str
    occurred_at: datetime


@dataclass(frozen=True)
class OrphanRepairEvent:
    """Emitted when the resilience layer synthesizes a ToolMessage for an
    orphan tool call found in state at model-call time.

    An orphan is an ``AIMessage(tool_calls=[...])`` whose ``tool_call_id``
    has no paired ``ToolMessage`` anywhere later in the message list.
    Orphans originate from crashed tool executions, pod kills between
    checkpoint writes, or manual state edits. The OpenAI Responses API
    rejects them with ``"No tool output found for function call"``.
    """

    tool_call_id: str
    tool_name: str
    ai_message_id: str | None
    repaired_at: datetime
