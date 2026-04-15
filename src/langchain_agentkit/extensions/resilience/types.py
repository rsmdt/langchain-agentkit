"""Typed events emitted by :class:`ResilienceExtension`."""

from __future__ import annotations

from dataclasses import dataclass
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
