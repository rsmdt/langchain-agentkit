"""History strategies for managing context window size.

Each strategy implements a ``transform`` method that takes a list of messages
and returns a (possibly shorter) list suitable for the LLM's context window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class HistoryStrategy(Protocol):
    """Protocol for custom history transformation strategies.

    Any object with a ``transform(messages) -> messages`` method satisfies
    this protocol and can be passed to ``HistoryExtension(strategy=...)``.
    """

    def transform(self, messages: list[Any]) -> list[Any]: ...


def _default_token_counter(message: Any) -> int:
    """Rough character-based token estimate (1 token ~ 4 chars)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return len(content) // 4
    return 0


def _is_system_message(message: Any) -> bool:
    """Check if a message is a SystemMessage without importing langchain."""
    return type(message).__name__ == "SystemMessage"


def _truncate_by_count(messages: list[Any], *, max_messages: int) -> list[Any]:
    """Keep the last *max_messages* messages, preserving a leading SystemMessage."""
    if len(messages) <= max_messages:
        return messages

    has_system = messages and _is_system_message(messages[0])
    if has_system:
        budget = max_messages - 1
        tail = messages[1:]
        return [messages[0]] + tail[-budget:] if budget > 0 else [messages[0]]

    return messages[-max_messages:]


def _truncate_by_tokens(
    messages: list[Any],
    *,
    max_tokens: int,
    token_counter: Callable[[Any], int] | None = None,
) -> list[Any]:
    """Keep the most recent messages that fit within *max_tokens*.

    Walks messages in reverse, accumulating token counts until the budget
    is exhausted.  A leading ``SystemMessage`` is always preserved with its
    tokens reserved first.
    """
    if not messages:
        return messages

    counter = token_counter or _default_token_counter
    budget = max_tokens
    has_system = _is_system_message(messages[0])

    if has_system:
        budget -= counter(messages[0])
        tail = messages[1:]
    else:
        tail = messages

    kept: list[Any] = []
    for msg in reversed(tail):
        cost = counter(msg)
        if budget - cost < 0 and kept:
            break
        kept.append(msg)
        budget -= cost

    kept.reverse()

    if has_system:
        return [messages[0]] + kept
    return kept
