"""HistoryExtension — rewrite graph state messages via a pluggable strategy.

This extension owns one thing: rewriting ``state["messages"]`` before
each LLM call. *What* the rewrite looks like is delegated entirely to
the strategy.

Three built-in strategies cover the common cases:

* :class:`CountStrategy` — roll-over: keep the last N messages.
* :class:`TokenStrategy` — roll-over: keep the tail within a token budget.
* :class:`CompactionStrategy` — collapse: when context fills, replace
  the conversation with one LLM-generated summary message.

Or supply any object implementing :class:`HistoryStrategy`.

Rewrites are persisted to graph state via :class:`ReplaceMessages`, so
the checkpointer stays in sync with what the LLM sees. There is no
"transparent" mode — every strategy is destructive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.history.state import ReplaceMessages

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_agentkit.extensions.history.strategies import HistoryStrategy


class HistoryExtension(Extension):
    """Rewrite ``state["messages"]`` before each LLM call via a strategy.

    Example::

        # Roll-over strategies
        HistoryExtension(strategy=CountStrategy(max_messages=50))
        HistoryExtension(strategy=TokenStrategy(max_tokens=4000))

        # Collapse-on-trigger strategy
        HistoryExtension(strategy=CompactionStrategy(reserve_tokens=16_384))

        # Custom
        class KeepLastTurn:
            async def transform(self, messages, *, runtime):
                return messages[-1:] if messages else []

        HistoryExtension(strategy=KeepLastTurn())
    """

    def __init__(self, *, strategy: HistoryStrategy) -> None:
        self._strategy = strategy

    @override
    async def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        strategy_setup = getattr(self._strategy, "setup", None)
        if callable(strategy_setup):
            await strategy_setup(llm_getter=kwargs.get("llm_getter"))

    async def wrap_model(
        self,
        *,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        original = list(state.get("messages", []))
        transformed = await self._strategy.transform(original, runtime=runtime)

        result = await handler({**state, "messages": transformed})

        if isinstance(result, dict):
            new_messages = result.get("messages", [])
            result["messages"] = ReplaceMessages(list(transformed) + list(new_messages))

        return result
