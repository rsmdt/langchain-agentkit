"""Tests for AgentLike delegation at runtime.

When the LLM delegates to an agent via the Agent tool, the delegation
code must handle both raw StateGraph objects and AgentLike objects.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_agentkit.composability import AgentLike, TeamAgent

# --- Helpers ---


def _make_agent_like(name: str, response: str = "done") -> AgentLike:
    class FakeAgent:
        def __init__(self, n, r):
            self._name = n
            self._response = r

        @property
        def name(self):
            return self._name

        @property
        def description(self):
            return f"{self._name} agent"

        async def ainvoke(self, input, config=None):
            return {
                "messages": [MagicMock(content=self._response, type="ai")],
                "sender": self._name,
            }

        async def astream(self, input, config=None):
            yield {"messages": [self._response]}

    return FakeAgent(name, response)


# --- _compile_or_resolve tests ---


class TestCompileOrResolve:
    """_compile_or_resolve should handle both raw graphs and AgentLike."""

    def test_returns_agent_like_directly(self):
        from langchain_agentkit.extensions.agents.tools.agent import _compile_or_resolve

        agent = _make_agent_like("researcher")
        result = _compile_or_resolve(agent, {}, None)

        assert result is agent

    def test_compiles_raw_graph(self):
        from langchain_agentkit.extensions.agents.tools.agent import _compile_or_resolve

        compiled = MagicMock()

        class FakeGraph:
            name = "researcher"
            tools_inherit = False
            nodes: dict = {}
            _compiled = compiled

            def compile(self, **kwargs):  # noqa: ANN003, ANN201, ARG002
                return self._compiled

        graph = FakeGraph()
        result = _compile_or_resolve(graph, {}, None)

        assert result is compiled

    def test_caches_compiled_graph(self):
        from langchain_agentkit.extensions.agents.tools.agent import _compile_or_resolve

        compiled = MagicMock()

        class FakeGraph:
            name = "researcher"
            tools_inherit = False
            nodes: dict = {}
            _compiled = compiled
            _compile_count = 0

            def compile(self, **kwargs):  # noqa: ANN003, ANN201, ARG002
                self._compile_count += 1
                return self._compiled

        graph = FakeGraph()
        cache: dict = {}

        result1 = _compile_or_resolve(graph, cache, None)
        result2 = _compile_or_resolve(graph, cache, None)

        assert result1 is result2
        assert graph._compile_count == 1  # Only compiled once

    def test_agent_like_not_cached(self):
        from langchain_agentkit.extensions.agents.tools.agent import _compile_or_resolve

        agent = _make_agent_like("researcher")
        cache = {}

        result = _compile_or_resolve(agent, cache, None)

        assert result is agent
        assert cache == {}  # AgentLike objects don't need caching


# --- TeamAgent delegation via Agent tool ---


class TestTeamAgentDelegation:
    """A TeamAgent should be delegatable via _compile_or_resolve."""

    def test_team_agent_returned_directly(self):
        from langchain_agentkit.extensions.agents.tools.agent import _compile_or_resolve

        lead = _make_agent_like("lead", "team result")
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("worker")])

        result = _compile_or_resolve(team, {}, None)

        # TeamAgent is AgentLike, so should be returned directly
        assert result is team

    @pytest.mark.asyncio
    async def test_team_agent_invocable(self):
        """TeamAgent.ainvoke should work when called by delegation code."""
        lead = _make_agent_like("lead", "team completed")
        team = TeamAgent(lead=lead, teammates=[_make_agent_like("worker")])

        result = await team.ainvoke({"messages": ["do work"]})

        assert result["sender"] == "lead"
