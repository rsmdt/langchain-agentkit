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

import logging
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.history.state import ReplaceMessages

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_agentkit.extensions.history.strategies import HistoryStrategy

_logger = logging.getLogger(__name__)


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
        self._warned_subgraph = False

    @override
    async def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        strategy_setup = getattr(self._strategy, "setup", None)
        if callable(strategy_setup):
            await strategy_setup(llm_getter=kwargs.get("llm_getter"))

    def _warn_if_subgraph(self, runtime: Any) -> None:
        """Warn once if this kit appears to run as a subgraph of another graph.

        History rewrites the shared ``messages`` channel; that rewrite escapes a
        subgraph boundary and breaks ``interrupt()`` resume in the parent. The
        recommended composition is to drop History from the embedded kit and let
        the outer graph own truncation (see ``docs/subgraph-composition.md``).

        Detection is a best-effort heuristic on LangGraph's checkpoint namespace:
        a nested run carries a parent segment (``"<parent>|<node>"``). It only
        ever suppresses the rewrite of a warning — never the rewrite itself — so
        a namespace-format change degrades to "no warning", not a failure.
        """
        if self._warned_subgraph:
            return
        try:
            config = getattr(runtime, "config", None) or {}
            namespace = (config.get("configurable") or {}).get("checkpoint_ns", "") or ""
        except Exception:  # noqa: BLE001 — advisory only, never break the run
            return
        if "|" in namespace:
            self._warned_subgraph = True
            _logger.warning(
                "HistoryExtension is running inside a subgraph. It rewrites the "
                "shared 'messages' channel, which breaks interrupt() resume across "
                "the subgraph boundary. Recommended: drop HistoryExtension from the "
                "embedded kit and apply truncation in the outer graph via the "
                "strategy's transform(). See docs/subgraph-composition.md."
            )

    async def wrap_model(
        self,
        *,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        self._warn_if_subgraph(runtime)
        original = list(state.get("messages", []))
        transformed = await self._strategy.transform(original, runtime=runtime)

        result = await handler({**state, "messages": transformed})

        if isinstance(result, dict):
            new_messages = result.get("messages", [])
            result["messages"] = ReplaceMessages(list(transformed) + list(new_messages))

        return result
