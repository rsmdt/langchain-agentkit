"""State schema and reducer for :class:`TurnBudgetExtension`.

The single field, ``_turn_budget_used``, counts completed model turns. It
uses an additive reducer so each turn's ``after_model`` hook can contribute a
``+1`` increment without reading-then-writing the absolute value — which keeps
the count correct under LangGraph's concurrent-update model.
"""

from __future__ import annotations

from typing import Annotated, TypedDict


def _accumulate(left: int | None, right: int | None) -> int:
    """Reducer that sums turn-count increments, treating unset as ``0``."""
    return (left or 0) + (right or 0)


class TurnBudgetState(TypedDict, total=False):
    """State mixin tracking how many model turns have completed.

    Internal bookkeeping owned entirely by :class:`TurnBudgetExtension`;
    handlers should not write this key directly.
    """

    _turn_budget_used: Annotated[int, _accumulate]
