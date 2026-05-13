"""History strategies — control how HistoryExtension rewrites messages.

A strategy is any object with an async ``transform(messages, *, runtime)``
method that returns the new message list. ``HistoryExtension`` always
persists the strategy's output to graph state via
:class:`ReplaceMessages` — the strategy decides what the new list is.

Two simple built-ins live here:

* :class:`CountStrategy` — keep the last N messages.
* :class:`TokenStrategy` — keep the tail within a token budget.

For LLM-driven summarizing compaction see
:class:`langchain_agentkit.extensions.history.CompactionStrategy`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class HistoryStrategy(Protocol):
    """Rewrites the message list before each LLM call.

    Required: ``async transform(messages, *, runtime) -> messages``.

    Optional (discovered via ``getattr``):

    * ``async setup(*, llm_getter)`` — receive a handle to the kit's
      main LLM at compile time. Strategies that need an LLM (e.g.
      :class:`CompactionStrategy`) implement this.
    * ``contribute_prompt() -> dict[str, str] | str | None`` — inject
      guidance into the agent's system prompt. Returned value follows
      :meth:`Extension.prompt` conventions.
    """

    async def transform(self, messages: list[Any], *, runtime: Any) -> list[Any]: ...


def _is_system_message(message: Any) -> bool:
    """Class-name check that avoids importing langchain at module load."""
    return type(message).__name__ == "SystemMessage"


def _default_token_counter(message: Any) -> int:
    """Rough character-based token estimate (1 token ~ 4 chars)."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return len(content) // 4
    return 0


class CountStrategy:
    """Keep the last ``max_messages`` messages.

    A leading ``SystemMessage`` is always preserved — it's the agent's
    persona, not part of the conversation history. When the budget is
    1 and a system message is present, only the system message is kept.
    """

    def __init__(self, *, max_messages: int) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self._max_messages = max_messages

    async def transform(self, messages: list[Any], *, runtime: Any) -> list[Any]:
        if len(messages) <= self._max_messages:
            return messages

        has_system = bool(messages) and _is_system_message(messages[0])
        if has_system:
            budget = self._max_messages - 1
            tail = messages[1:]
            return [messages[0], *tail[-budget:]] if budget > 0 else [messages[0]]

        return messages[-self._max_messages :]


class TokenStrategy:
    """Keep the most recent messages that fit within ``max_tokens``.

    Walks from the newest message backward, accumulating per-message
    token counts until the budget is exhausted. A leading
    ``SystemMessage`` is preserved with its tokens reserved first.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        token_counter: Callable[[Any], int] | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_tokens = max_tokens
        self._counter = token_counter or _default_token_counter

    async def transform(self, messages: list[Any], *, runtime: Any) -> list[Any]:
        if not messages:
            return messages

        budget = self._max_tokens
        has_system = _is_system_message(messages[0])

        if has_system:
            budget -= self._counter(messages[0])
            tail = messages[1:]
        else:
            tail = messages

        kept: list[Any] = []
        for msg in reversed(tail):
            cost = self._counter(msg)
            if budget - cost < 0 and kept:
                break
            kept.append(msg)
            budget -= cost

        kept.reverse()
        if has_system:
            return [messages[0], *kept]
        return kept
