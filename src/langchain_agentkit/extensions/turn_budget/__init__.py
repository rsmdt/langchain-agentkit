"""TurnBudgetExtension — graceful, in-graph cap on ReAct loop length.

Bound an agent to ``max_turns`` model calls, surface a per-turn "N turns
left" reminder in the system prompt, and stop the loop cleanly (no
``GraphRecursionError``) with a final wrap-up turn when the budget is spent.

See :class:`TurnBudgetExtension` for the full contract.
"""

from langchain_agentkit.extensions.turn_budget.extension import TurnBudgetExtension
from langchain_agentkit.extensions.turn_budget.state import TurnBudgetState

__all__ = ["TurnBudgetExtension", "TurnBudgetState"]
