"""Message filter utilities for team-tagged conversations.

Teammate-internal messages (HumanMessage input, AIMessage reasoning,
ToolMessage tool results) are stored in the shared ``state["messages"]``
channel alongside the lead's own conversation.  Each teammate message is
tagged via ``additional_kwargs["team"] = {"member": <name>}`` so
consumers can filter them.

All team metadata lives under a single ``"team"`` key in
``additional_kwargs``.  Presence of the key means "teammate-internal,
hide from the lead's LLM."

**The tag is authoritative.**  Do not filter by ``AIMessage.name`` alone
--- ``ToolMessage.name`` holds the tool name (not the teammate), and
filtering only AIMessages by name orphans their ToolMessage results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


TEAM_KEY = "team"
"""Root key in ``additional_kwargs`` for all team metadata."""


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _team_meta(msg: BaseMessage) -> dict[str, Any]:
    """Read the ``additional_kwargs["team"]`` dict, or empty dict if absent."""
    kwargs: dict[str, Any] = getattr(msg, "additional_kwargs", {}) or {}
    return kwargs.get(TEAM_KEY) or {}


def is_team_tagged(msg: BaseMessage) -> bool:
    """Return True if the message was produced inside a teammate's conversation.

    The check is simply whether ``additional_kwargs["team"]`` exists.
    """
    kwargs: dict[str, Any] = getattr(msg, "additional_kwargs", {}) or {}
    return TEAM_KEY in kwargs


def team_member_of(msg: BaseMessage) -> str | None:
    """Return the teammate name the message belongs to, or None if untagged."""
    value = _team_meta(msg).get("member")
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_out_team_messages(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Return messages excluding all team-tagged ones.

    Used by the lead's ``wrap_model`` hook so the lead's LLM sees only
    its own conversation --- the team-tagged messages stay in the audit
    trail but are invisible to the lead model call.
    """
    return [m for m in messages if not is_team_tagged(m)]


def filter_team_messages(
    messages: list[BaseMessage],
    member_name: str,
) -> list[BaseMessage]:
    """Return only messages belonging to one specific teammate.

    Used during rehydration to reconstruct a teammate's prior history
    from the checkpointed ``state["messages"]``.
    """
    return [m for m in messages if team_member_of(m) == member_name]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def tag_message(msg: BaseMessage, member_name: str) -> None:
    """Tag a message in place as belonging to a specific teammate.

    Sets ``additional_kwargs["team"]["member"]`` (the canonical filter
    key) and, for ``AIMessage``s without a ``name``, also sets ``name``
    for observability / OpenAI-native author rendering.  Safe to call
    repeatedly --- idempotent.
    """
    from langchain_core.messages import AIMessage

    msg.additional_kwargs = dict(msg.additional_kwargs or {})
    meta = dict(msg.additional_kwargs.get(TEAM_KEY) or {})
    meta["member"] = member_name
    msg.additional_kwargs[TEAM_KEY] = meta
    if isinstance(msg, AIMessage) and not msg.name:
        msg.name = member_name
