"""HistoryExtension — context window management via pluggable strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.history.strategies import (
    _truncate_by_count,
    _truncate_by_tokens,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_agentkit.extensions.history.strategies import HistoryStrategy


class HistoryExtension(Extension):
    """Extension that manages message history before each LLM call.

    Truncates messages to fit the configured window.  The LLM only sees
    the truncated window, and dropped messages are replaced in graph
    state via a custom reducer so the checkpointer stays lean.

    Built-in strategies::

        # Keep the last 50 messages
        HistoryExtension(strategy="count", max_messages=50)

        # Keep messages within a token budget
        HistoryExtension(strategy="tokens", max_tokens=4000)

        # With a custom token counter
        HistoryExtension(strategy="tokens", max_tokens=4000, token_counter=my_fn)

    Custom strategy (any object with ``transform(messages) -> messages``)::

        HistoryExtension(strategy=MySummarizationStrategy())
    """

    def __init__(
        self,
        *,
        strategy: Literal["count", "tokens"] | HistoryStrategy,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        token_counter: Callable[[Any], int] | None = None,
    ) -> None:
        if isinstance(strategy, str):
            self._init_builtin(strategy, max_messages, max_tokens, token_counter)
        else:
            self._custom_strategy = strategy
            self._builtin: str | None = None

    def _init_builtin(
        self,
        strategy: str,
        max_messages: int | None,
        max_tokens: int | None,
        token_counter: Callable[[Any], int] | None,
    ) -> None:
        if strategy == "count":
            if max_messages is None:
                raise ValueError("max_messages is required for strategy='count'")
            if max_messages < 1:
                raise ValueError("max_messages must be >= 1")
            self._builtin = "count"
            self._max_messages = max_messages
        elif strategy == "tokens":
            if max_tokens is None:
                raise ValueError("max_tokens is required for strategy='tokens'")
            if max_tokens < 1:
                raise ValueError("max_tokens must be >= 1")
            self._builtin = "tokens"
            self._max_tokens = max_tokens
            self._token_counter = token_counter
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'count', 'tokens', "
                f"or pass a custom HistoryStrategy object."
            )

    async def wrap_model(self, *, state: dict[str, Any], handler: Any, runtime: Any) -> Any:
        """Truncate messages, call the LLM, and replace graph state.

        1. Compute the truncated window from ``state["messages"]``.
        2. Pass the truncated state to the inner handler (LLM sees only
           the window).
        3. Return ``ReplaceMessages(kept + response)`` so the custom
           reducer replaces the full list in one operation.
        """
        from langchain_agentkit.extensions.history.state import ReplaceMessages

        original = list(state.get("messages", []))
        kept = self._transform(original)

        # LLM sees only the truncated window
        truncated_state = {**state, "messages": kept}
        result = await handler(truncated_state)

        # Replace the entire messages list: kept window + new response
        if isinstance(result, dict):
            new_messages = result.get("messages", [])
            result["messages"] = ReplaceMessages(kept + list(new_messages))

        return result

    def _transform(self, messages: list[Any]) -> list[Any]:
        """Apply the strategy to transform the message list."""
        if self._builtin == "count":
            return _truncate_by_count(messages, max_messages=self._max_messages)
        if self._builtin == "tokens":
            return _truncate_by_tokens(
                messages,
                max_tokens=self._max_tokens,
                token_counter=self._token_counter,
            )
        return self._custom_strategy.transform(messages)
