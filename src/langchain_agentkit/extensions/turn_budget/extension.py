"""TurnBudgetExtension — a graceful, in-graph cap on ReAct loop length.

LangGraph's native ``recursion_limit`` is a *hard* ceiling: when the loop
exceeds it, LangGraph raises ``GraphRecursionError`` mid-run and whatever
partial state existed is lost to the caller. This extension enforces a turn
budget from *inside* the graph instead, and stops the loop **gracefully**:

* It states the total budget once in the (cacheable) system prompt, and every
  step contributes a reminder (appended to the tail of the conversation)
  stating how many turns remain before a final answer is required.
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

_BUDGET_PROMPT = (
    "There are a total of {max_turns} turns available. Provide a final answer "
    "within that budget, ideally as early as possible."
)
_FINAL_REMINDER = "This is your last turn — you must provide a final answer now."


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
        # ``_turn_budget_used`` holds the turns completed *before* this call,
        # so the turn about to run is ``used + 1``.
        budget = _BUDGET_PROMPT.format(max_turns=self._max_turns)
        used = state.get("_turn_budget_used", 0) or 0
        remaining = self._max_turns - (used + 1)
        if remaining <= 0:
            return {"prompt": budget, "reminder": _FINAL_REMINDER}
        verb, noun = ("is", "turn") if remaining == 1 else ("are", "turns")
        reminder = f"There {verb} {remaining} {noun} left before you must provide a final answer."
        return {"prompt": budget, "reminder": reminder}

    async def after_model(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        # Returns either a ``+1`` increment or a ``jump_to: end`` — never both,
        # because an ``after_model`` update carrying ``jump_to`` discards its
        # sibling keys in the graph builder.
        used = state.get("_turn_budget_used", 0) or 0
        if used + 1 >= self._max_turns:
            return {"jump_to": "end"}
        return {"_turn_budget_used": 1}
