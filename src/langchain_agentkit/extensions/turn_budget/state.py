"""State schema for :class:`TurnBudgetExtension`.

The single field, ``budget``, counts the model turns completed in the current
run. It is private (kept out of the graph's input and output schema) and
plain-overwrite: ``before_run`` resets it to ``0`` and ``after_model`` writes
the absolute count, so a run never inherits turns spent by an earlier run on
the same thread.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_agentkit.composition.state import PrivateStateAttr


class TurnBudgetState(TypedDict, total=False):
    """State mixin tracking how many model turns the current run has used.

    Internal bookkeeping owned entirely by :class:`TurnBudgetExtension`;
    handlers should not write this key directly.
    """

    budget: Annotated[int, PrivateStateAttr]
