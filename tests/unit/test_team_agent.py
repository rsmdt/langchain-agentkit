"""Tests for TeamAgent (SocietyOfMindAgent pattern).

TeamAgent wraps a lead agent + teammates as a single AgentLike.
The inner team runs to completion, then the result is returned.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent


# --- Helpers ---


def _make_agent_like(name: str, response: str = "done") -> AgentLike:
    """Create a simple AgentLike that returns a fixed response."""

    class FakeAgent:
        def __init__(self, n, r):
            self._name = n
            self._response = r

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return f"{self._name} agent"

        async def ainvoke(self, input, config=None):
            return {"messages": [self._response], "sender": self._name}

        async def astream(self, input, config=None):
            yield {"messages": [self._response], "sender": self._name}

    return FakeAgent(name, response)


# --- Protocol conformance ---


class TestTeamAgentIsAgentLike:
    """TeamAgent must satisfy the AgentLike protocol."""

    def test_is_agent_like(self):
        lead = _make_agent_like("lead", "team result")
        teammates = [_make_agent_like("worker")]

        team = TeamAgent(lead=lead, teammates=teammates)

        assert isinstance(team, AgentLike)

    def test_has_name(self):
        lead = _make_agent_like("project_lead")
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("w")])

        assert team.name == "project_lead"

    def test_has_description(self):
        lead = _make_agent_like("project_lead")
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("w")])

        assert team.description == "project_lead agent"


# --- Core behavior ---


class TestTeamAgentInvoke:
    """TeamAgent.ainvoke runs the lead to completion and returns the result."""

    @pytest.mark.asyncio
    async def test_returns_lead_result(self):
        lead = _make_agent_like("lead", "team completed the work")
        teammates = [_make_agent_like("researcher")]

        team = TeamAgent(lead=lead, teammates=teammates)
        result = await team.ainvoke({"messages": ["do research"]})

        assert result["messages"] == ["team completed the work"]
        assert result["sender"] == "lead"

    @pytest.mark.asyncio
    async def test_passes_input_to_lead(self):
        class TrackingAgent:
            @property
            def name(self):
                return "lead"

            @property
            def description(self):
                return ""

            async def ainvoke(self, input, config=None):
                self.received_input = input
                return {"messages": ["ok"], "sender": "lead"}

            async def astream(self, input, config=None):
                yield {"messages": ["ok"]}

        lead = TrackingAgent()
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("w")])

        await team.ainvoke({"messages": ["hello team"]})

        assert lead.received_input["messages"] == ["hello team"]

    @pytest.mark.asyncio
    async def test_passes_config_to_lead(self):
        class TrackingAgent:
            @property
            def name(self):
                return "lead"

            @property
            def description(self):
                return ""

            async def ainvoke(self, input, config=None):
                self.received_config = config
                return {"messages": ["ok"], "sender": "lead"}

            async def astream(self, input, config=None):
                yield {"messages": ["ok"]}

        lead = TrackingAgent()
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("w")])
        config = {"configurable": {"thread_id": "123"}}

        await team.ainvoke({"messages": ["hello"]}, config=config)

        assert lead.received_config == config


# --- Teammate exposure ---


class TestTeamAgentTeammates:
    """TeamAgent exposes teammates for introspection."""

    def test_teammates_accessible(self):
        lead = _make_agent_like("lead")
        w1 = _make_agent_like("worker_1")
        w2 = _make_agent_like("worker_2")

        team = TeamAgent(lead=lead, teammates=[w1, w2])

        assert len(team.teammates) == 2
        assert team.teammates[0].name == "worker_1"
        assert team.teammates[1].name == "worker_2"

    def test_lead_accessible(self):
        lead = _make_agent_like("lead")
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("w")])

        assert team.lead is lead


# --- Nesting ---


class TestTeamAgentNesting:
    """TeamAgent can be nested — a teammate can be a TeamAgent."""

    @pytest.mark.asyncio
    async def test_nested_team_as_teammate(self):
        inner_lead = _make_agent_like("inner_lead", "inner result")
        inner_worker = _make_agent_like("inner_worker")
        inner_team = TeamAgent(lead=inner_lead, teammates=[inner_worker])

        outer_lead = _make_agent_like("outer_lead", "outer result")
        outer_team = TeamAgent(lead=outer_lead, teammates=[inner_team])

        # The outer team should work — inner_team is a valid AgentLike
        assert isinstance(inner_team, AgentLike)
        assert outer_team.teammates[0].name == "inner_lead"

    def test_nested_team_is_agent_like(self):
        inner = TeamAgent(
            lead=_make_agent_like("inner"),
            teammates=[_make_agent_like("iw")],
        )
        assert isinstance(inner, AgentLike)


# --- Edge cases ---


class TestTeamAgentEdgeCases:
    """Edge cases and error handling."""

    def test_requires_at_least_one_teammate(self):
        lead = _make_agent_like("lead")

        with pytest.raises(ValueError, match="at least one teammate"):
            TeamAgent(lead=lead, teammates=[])

    def test_lead_cannot_be_none(self):
        with pytest.raises((TypeError, ValueError)):
            TeamAgent(lead=None, teammates=[_make_agent_like("w")])
