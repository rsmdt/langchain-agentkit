"""Tests for subagent turn-budget inheritance.

Covers the decision point (:func:`resolve_inherited_budget`,
:func:`detect_parent_soft_budget`), the outer-budget surface
(``AgentKit.max_turns`` exposed through ``run_extension_setup``), and the
application point (``build_ephemeral_graph`` composing an inherited
``TurnBudgetExtension`` into a runnable graph).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from langchain_agentkit._internal.graph_builder import build_ephemeral_graph
from langchain_agentkit.composition.agent_kit import AgentKit, run_extension_setup
from langchain_agentkit.extensions.agents.budget import (
    detect_parent_soft_budget,
    resolve_inherited_budget,
)
from langchain_agentkit.extensions.agents.tools.agent import _run_delegation
from langchain_agentkit.extensions.turn_budget import TurnBudgetExtension


class TestResolveInheritedBudget:
    """An own value wins; otherwise the parent's hard and/or soft budgets apply."""

    def test_own_value_wins_no_inheritance(self):
        """An explicit own max_turns is a hard ceiling and suppresses inheritance."""
        hard, exts = resolve_inherited_budget(5, parent_hard=20, parent_soft=10)
        assert hard == 5
        assert exts == []

    def test_inherits_hard_only(self):
        """Unset own + parent hard ceiling → inherit the ceiling, no soft extension."""
        hard, exts = resolve_inherited_budget(None, parent_hard=20, parent_soft=None)
        assert hard == 20
        assert exts == []

    def test_inherits_soft_only(self):
        """Unset own + soft-only → inject TurnBudgetExtension; backstop lifted to soft.

        The hard backstop is set to the soft value so the soft budget is always
        honored even when it exceeds LangGraph's default recursion limit.
        """
        hard, exts = resolve_inherited_budget(None, parent_hard=None, parent_soft=10)
        assert hard == 10
        assert len(exts) == 1
        assert isinstance(exts[0], TurnBudgetExtension)
        assert exts[0].max_turns == 10

    def test_inherits_both(self):
        """Parent with both → subagent gets soft primary + hard backstop."""
        hard, exts = resolve_inherited_budget(None, parent_hard=20, parent_soft=10)
        assert hard == 20
        assert len(exts) == 1
        assert exts[0].max_turns == 10

    def test_hard_below_soft_is_lifted(self):
        """Misconfigured parent (hard < soft) → backstop lifted to soft, soft fires first."""
        hard, exts = resolve_inherited_budget(None, parent_hard=5, parent_soft=10)
        assert hard == 10
        assert exts[0].max_turns == 10

    def test_inherits_nothing(self):
        """No own, no parent budget → LangGraph default (None, [])."""
        hard, exts = resolve_inherited_budget(None, parent_hard=None, parent_soft=None)
        assert hard is None
        assert exts == []


class TestDetectParentSoftBudget:
    """Soft budget is read from a TurnBudgetExtension in the parent stack."""

    def test_present(self):
        assert detect_parent_soft_budget([TurnBudgetExtension(max_turns=8)]) == 8

    def test_absent(self):
        assert detect_parent_soft_budget([]) is None


class TestAgentKitMaxTurns:
    """``AgentKit.max_turns`` is stored and surfaced to extension setup."""

    def test_property_roundtrip(self):
        assert AgentKit(max_turns=9).max_turns == 9
        assert AgentKit().max_turns is None

    @pytest.mark.asyncio
    async def test_setup_receives_max_turns(self):
        """run_extension_setup hands max_turns to an extension that accepts it."""
        captured: dict[str, object] = {}

        class _Capture:
            def setup(self, *, max_turns=None, **_):
                captured["max_turns"] = max_turns

            @property
            def tools(self):
                return []

        await run_extension_setup(AgentKit(extensions=[_Capture()], max_turns=4))
        assert captured["max_turns"] == 4


class TestEphemeralGraphComposesInheritedExtensions:
    """An inherited TurnBudgetExtension composes into a runnable ephemeral graph."""

    @pytest.mark.asyncio
    async def test_turn_budget_extension_injected(self):
        """The soft budget surfaces in the system prompt of the ephemeral agent."""
        seen_prompts: list[str] = []

        async def _capture(messages):
            seen_prompts.append(messages[0].content)
            return AIMessage(content="done")

        bound = MagicMock()
        bound.ainvoke = AsyncMock(side_effect=_capture)
        llm = MagicMock()
        llm.bind_tools = MagicMock(return_value=bound)
        llm.ainvoke = AsyncMock(side_effect=_capture)

        graph = build_ephemeral_graph(
            name="bounded",
            llm=llm,
            prompt="you are bounded",
            extensions=[TurnBudgetExtension(max_turns=3)],
        )
        await graph.ainvoke({"messages": [HumanMessage(content="go")]})

        # TurnBudgetExtension states the budget in the (system) prompt.
        assert any("3 turns" in p for p in seen_prompts)


class TestTurnLimitErrorMessage:
    """A hard-ceiling hit yields a distinct, non-generic error ToolMessage."""

    @pytest.mark.asyncio
    async def test_recursion_error_is_distinct(self):
        """GraphRecursionError → a turn-limit message naming the budget, not a crash."""
        compiled = MagicMock()
        compiled.ainvoke = AsyncMock(side_effect=GraphRecursionError("limit"))

        command = await _run_delegation(
            compiled=compiled,
            scoped_state={"messages": []},
            agent="worker",
            timeout=30.0,
            tool_call_id="call-1",
            max_turns=6,
        )

        msg = command.update["messages"][0]
        assert msg.status == "error"
        assert "turn limit" in msg.content
        assert "6 turns" in msg.content
        assert "internal error" not in msg.content
