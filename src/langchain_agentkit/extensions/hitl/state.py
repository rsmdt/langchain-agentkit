"""State mixin for the HITL approval flow.

The approval node records non-executing decisions (reject / respond) into
``hitl_decisions`` keyed by tool-call id. The ``wrap_tool`` hook reads the
channel in the same step and substitutes the corresponding ``ToolMessage``
instead of executing the tool — preserving the one-``ToolMessage``-per-tool-call
invariant without relying on ToolNode skip behaviour.
"""

from __future__ import annotations

from typing import Any, TypedDict


class HITLState(TypedDict, total=False):
    """Per-step decisions for tool calls the human did not approve outright.

    Maps ``tool_call_id`` to a decision dict (``{"type": "reject"|"respond",
    "message": str}``). Overwritten every tool step by the approval node, so a
    stale entry never leaks into a later step.
    """

    hitl_decisions: dict[str, Any]
