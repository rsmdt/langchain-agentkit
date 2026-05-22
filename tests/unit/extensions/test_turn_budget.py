# ruff: noqa: N805
"""Tests for TurnBudgetExtension — graceful, in-graph turn capping."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_agentkit import Agent
from langchain_agentkit.extensions.turn_budget import TurnBudgetExtension, TurnBudgetState
from langchain_agentkit.extensions.turn_budget.state import _accumulate


def _make_llm() -> MagicMock:
    """A mock LLM; the test handlers return canned messages and never call it."""
    mock = MagicMock()
    mock.bind_tools = MagicMock(return_value=mock)
    mock.ainvoke = MagicMock(return_value=AIMessage(content="response"))
    return mock


class TestConstruction:
    def test_stores_max_turns(self):
        assert TurnBudgetExtension(max_turns=5)._max_turns == 5

    @pytest.mark.parametrize("bad", [0, -1, -10])
    def test_rejects_non_positive_max_turns(self, bad):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            TurnBudgetExtension(max_turns=bad)

    def test_state_schema_is_turn_budget_state(self):
        assert TurnBudgetExtension(max_turns=3).state_schema is TurnBudgetState


class TestReducer:
    def test_treats_unset_as_zero(self):
        assert _accumulate(None, 1) == 1

    def test_sums_increments(self):
        assert _accumulate(2, 1) == 3

    def test_handles_unset_increment(self):
        assert _accumulate(1, None) == 1


class TestHookDiscovery:
    def test_after_model_discovered_as_named_hook(self):
        assert ("after", "model") in TurnBudgetExtension._get_named_hooks()


class TestReminder:
    def test_system_prompt_states_total_budget(self):
        ext = TurnBudgetExtension(max_turns=7)
        # The total budget is static -> system prompt (same on every turn).
        assert ext.prompt({})["prompt"] == (
            "There are a total of 7 turns available. Provide a final answer "
            "within that budget, ideally as early as possible."
        )
        assert ext.prompt({"_turn_budget_used": 6})["prompt"] == (
            "There are a total of 7 turns available. Provide a final answer "
            "within that budget, ideally as early as possible."
        )

    def test_first_turn_states_turns_remaining(self):
        ext = TurnBudgetExtension(max_turns=3)
        reminder = ext.prompt({})["reminder"]
        assert reminder == "There are 2 turns left before you must provide a final answer."

    def test_singular_turn_remaining(self):
        ext = TurnBudgetExtension(max_turns=3)
        reminder = ext.prompt({"_turn_budget_used": 1})["reminder"]
        assert reminder == "There is 1 turn left before you must provide a final answer."

    def test_final_turn_switches_to_wrap_up(self):
        ext = TurnBudgetExtension(max_turns=3)
        reminder = ext.prompt({"_turn_budget_used": 2})["reminder"]
        assert reminder == "This is your last turn — you must provide a final answer now."

    def test_max_turns_one_is_immediately_final(self):
        ext = TurnBudgetExtension(max_turns=1)
        reminder = ext.prompt({})["reminder"]
        assert reminder == "This is your last turn — you must provide a final answer now."


class TestAfterModelGate:
    async def test_increments_on_non_final_turn(self):
        ext = TurnBudgetExtension(max_turns=3)
        assert await ext.after_model(state={}, runtime=None) == {"_turn_budget_used": 1}
        assert await ext.after_model(state={"_turn_budget_used": 1}, runtime=None) == {
            "_turn_budget_used": 1
        }

    async def test_ends_loop_on_final_turn(self):
        ext = TurnBudgetExtension(max_turns=3)
        assert await ext.after_model(state={"_turn_budget_used": 2}, runtime=None) == {
            "jump_to": "end"
        }

    async def test_max_turns_one_ends_after_first_turn(self):
        ext = TurnBudgetExtension(max_turns=1)
        assert await ext.after_model(state={}, runtime=None) == {"jump_to": "end"}

    async def test_never_returns_both_increment_and_jump(self):
        """jump_to and the counter must never share one dict (builder drops siblings)."""
        ext = TurnBudgetExtension(max_turns=2)
        for used in range(5):
            result = await ext.after_model(state={"_turn_budget_used": used}, runtime=None)
            assert not ("jump_to" in result and "_turn_budget_used" in result)


class TestGraphIntegration:
    """Drive a real graph whose handler loops forever without a budget."""

    async def test_loop_stops_at_budget_without_recursion_error(self):
        calls = {"n": 0}

        @tool
        def noop(x: str) -> str:
            """Echo helper used only to give the graph an executable tool."""
            return "ok"

        class LoopingAgent(Agent):
            model = _make_llm()
            tools = [noop]
            extensions = [TurnBudgetExtension(max_turns=3)]

            async def handler(state, *, llm):
                calls["n"] += 1
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": f"c{calls['n']}",
                                    "name": "noop",
                                    "args": {"x": "a"},
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ],
                    "sender": "looping_agent",
                }

        compiled = await LoopingAgent().compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="go")]})

        # Exactly max_turns model calls — would hit GraphRecursionError (default
        # limit 25) if the budget had not stopped the loop gracefully first.
        assert calls["n"] == 3
        # Incremented on turns 1 and 2; the final turn ends without incrementing.
        assert result["_turn_budget_used"] == 2

    async def test_early_finish_does_not_trip_budget(self):
        """A response with no tool calls ends the loop before the budget fires."""
        calls = {"n": 0}

        @tool
        def noop(x: str) -> str:
            """Echo helper."""
            return "ok"

        class QuickAgent(Agent):
            model = _make_llm()
            tools = [noop]
            extensions = [TurnBudgetExtension(max_turns=10)]

            async def handler(state, *, llm):
                calls["n"] += 1
                return {"messages": [AIMessage(content="done")], "sender": "quick_agent"}

        compiled = await QuickAgent().compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="go")]})

        assert calls["n"] == 1
        assert result["messages"][-1].content == "done"
