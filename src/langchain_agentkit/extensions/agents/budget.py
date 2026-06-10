"""Turn-budget inheritance for subagents.

A subagent's turn budget resolves in this order:

1. **Its own** ``max_turns`` (from ``AgentConfig`` / ``maxTurns`` frontmatter).
   When set, it is a hard recursion ceiling and wins outright — the documented
   ``AgentConfig.max_turns`` semantics are preserved.
2. **Inherited from the parent** when the subagent declares none. The parent
   may carry a hard ceiling (``AgentKit.max_turns``), a soft budget
   (a ``TurnBudgetExtension`` in its stack), or both — each propagates so the
   subagent mirrors the parent's bounding behavior.
3. **LangGraph default** (25) when neither the subagent nor the parent sets one.

Shared by ``AgentsExtension`` and ``TeamExtension`` so delegation and team
ephemeral teammates bound their loops identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_agentkit.composition.extension import Extension


def detect_parent_soft_budget(extensions: list[Extension]) -> int | None:
    """Return the parent's soft turn budget, if a ``TurnBudgetExtension`` is present.

    Scans the parent's extension stack for a ``TurnBudgetExtension`` and reads
    its configured ``max_turns``. Returns ``None`` when no soft budget is set.
    """
    from langchain_agentkit.extensions.turn_budget import TurnBudgetExtension

    for ext in extensions:
        if isinstance(ext, TurnBudgetExtension):
            return ext.max_turns
    return None


def resolve_inherited_budget(
    own_max_turns: int | None,
    parent_hard: int | None,
    parent_soft: int | None,
) -> tuple[int | None, list[Any]]:
    """Resolve a subagent's effective turn budget.

    Returns ``(hard_max_turns, extra_extensions)`` to hand to
    ``build_ephemeral_graph``:

    - ``hard_max_turns`` caps ``recursion_limit`` (``max_turns * 2``); ``None``
      leaves LangGraph's default in place.
    - ``extra_extensions`` carries a fresh ``TurnBudgetExtension`` when a soft
      budget is inherited, so the subagent stops gracefully rather than
      crashing on the ceiling.

    An explicit ``own_max_turns`` is a hard ceiling and wins; no inheritance
    applies. Otherwise the parent's hard and/or soft budgets propagate.

    When a soft budget is inherited, the hard backstop is lifted to at least
    the soft value. The hard ceiling has no graceful-exit logic of its own —
    only the soft ``TurnBudgetExtension`` wraps up cleanly — so the backstop
    must never sit *below* the soft budget, or it would crash the loop before
    the soft budget's graceful final turn (and before LangGraph's default
    recursion limit, when no hard ceiling was inherited at all).
    """
    if own_max_turns is not None:
        return own_max_turns, []

    hard = parent_hard
    extra_extensions: list[Any] = []
    if parent_soft is not None:
        from langchain_agentkit.extensions.turn_budget import TurnBudgetExtension

        extra_extensions.append(TurnBudgetExtension(max_turns=parent_soft))
        hard = parent_soft if hard is None else max(hard, parent_soft)
    return hard, extra_extensions
