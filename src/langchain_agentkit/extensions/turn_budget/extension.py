"""TurnBudgetExtension — a graceful, in-graph cap on ReAct loop length.

LangGraph's native ``recursion_limit`` is a *hard* ceiling: when the loop
exceeds it, LangGraph raises ``GraphRecursionError`` mid-run and whatever
partial state existed is lost to the caller. This extension enforces a turn
budget from *inside* the graph instead, and stops the loop **gracefully**:

* Every step it contributes a reminder (appended to the tail of the
  conversation) telling the model which turn it is on and how many remain.
* On the final allowed turn it switches that reminder to a wrap-up
  instruction and, once the turn's model call returns, routes the loop to
  ``END`` — so the run finishes with a normal assistant message instead of an
  exception.

A "turn" is one model call (one handler invocation). ``max_turns=10`` permits
at most ten model calls. The count lives in graph state
(``_turn_budget_used``), so it is checkpointed and survives resumption.

The budget is a *ceiling*, not a floor: if the model finishes early (a
response with no tool calls), the loop ends normally and the budget never
fires.

**Tool access on the final turn.** The framework contract puts
``llm.bind_tools()`` in the handler's hands, so this extension cannot
physically unbind tools for the final call. Instead it guarantees no tool
*runs* on the final turn — the loop exits before the tools node — and the
reminder instructs the model accordingly. If the model still emits tool calls
on its final turn, they remain on the trailing assistant message unanswered;
that is a benign terminal state.

This extension is self-contained: it makes no core changes and does not set
LangGraph's ``recursion_limit``. If you also want a hard backstop against
runaway loops, set ``Agent.max_turns`` (or pass ``recursion_limit``) to a
value above this budget.
"""

from __future__ import annotations

from typing import Any, override

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.turn_budget.state import TurnBudgetState

_REMINDER_TEMPLATE = (
    "You are on step {current} of {max_turns} — each model response or tool "
    "call counts as one step. {remaining} step{plural} remain after this one "
    "before the run ends automatically. Pace your work so you can deliver a "
    "complete answer within the budget."
)

_FINAL_TEMPLATE = (
    "This is your final step ({max_turns} of {max_turns}) — each model "
    "response or tool call counts as one step. The run ends immediately after "
    "this response, so any tool calls you make now will NOT be executed. Give "
    "your best, complete answer using what you already have."
)


class TurnBudgetExtension(Extension):
    """Bound the ReAct loop to ``max_turns`` model calls, gracefully.

    Example::

        from langchain_agentkit import Agent
        from langchain_agentkit.extensions.turn_budget import TurnBudgetExtension

        class Researcher(Agent):
            model = ChatOpenAI(model="gpt-4o")
            extensions = [TurnBudgetExtension(max_turns=10)]

            async def handler(state, *, llm, tools, prompt):
                ...

    Args:
        max_turns: Maximum number of model calls (turns) the loop may take.
            Must be ``>= 1``.
    """

    def __init__(self, *, max_turns: int) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self._max_turns = max_turns

    @property
    @override
    def state_schema(self) -> type:
        return TurnBudgetState

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: Any | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> dict[str, str]:
        """Contribute the per-turn budget reminder.

        ``_turn_budget_used`` holds the number of turns completed *before*
        this call, so the turn about to run is ``used + 1``.
        """
        used = state.get("_turn_budget_used", 0) or 0
        current = used + 1
        if current >= self._max_turns:
            return {"reminder": _FINAL_TEMPLATE.format(max_turns=self._max_turns)}
        remaining = self._max_turns - current
        return {
            "reminder": _REMINDER_TEMPLATE.format(
                current=current,
                max_turns=self._max_turns,
                remaining=remaining,
                plural="" if remaining == 1 else "s",
            )
        }

    async def after_model(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        """Count the completed turn, or end the loop if the budget is spent.

        Returns either a ``+1`` increment (normal turn) or a ``jump_to: end``
        (final turn) — never both, because an ``after_model`` update carrying
        ``jump_to`` discards its sibling keys in the graph builder.
        """
        used = state.get("_turn_budget_used", 0) or 0
        if used + 1 >= self._max_turns:
            return {"jump_to": "end"}
        return {"_turn_budget_used": 1}
