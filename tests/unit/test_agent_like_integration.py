"""Tests for AgentLike integration into AgentExtension and TeamExtension."""

from __future__ import annotations

import pytest

from langchain_agentkit.composability import AgentLike, CompiledAgent, wrap_if_needed


# --- Helpers ---

def _make_mock_agent(name: str, description: str = "") -> object:
    """Create a mock object that looks like a compiled agent graph."""
    from unittest.mock import MagicMock

    graph = MagicMock()
    graph.name = name
    graph.description = description
    graph.tools_inherit = False
    return graph


def _make_agent_like(name: str, description: str = "") -> AgentLike:
    """Create an AgentLike object."""

    class FakeAgent:
        def __init__(self, n, d):
            self._name = n
            self._desc = d

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._desc

        async def ainvoke(self, input, config=None):
            return {"messages": [f"response from {self._name}"]}

        async def astream(self, input, config=None):
            yield {"messages": [f"stream from {self._name}"]}

    return FakeAgent(name, description)


# --- validate_agent_list tests ---


class TestValidateAgentList:
    """validate_agent_list should accept both raw graphs and AgentLike objects."""

    def test_accepts_raw_graphs(self):
        from langchain_agentkit.extensions import validate_agent_list

        agents = [_make_mock_agent("a"), _make_mock_agent("b")]
        result = validate_agent_list(agents)

        assert "a" in result
        assert "b" in result

    def test_accepts_agent_like_objects(self):
        from langchain_agentkit.extensions import validate_agent_list

        agents = [_make_agent_like("agent_a", "desc a"), _make_agent_like("agent_b", "desc b")]
        result = validate_agent_list(agents)

        assert "agent_a" in result
        assert "agent_b" in result

    def test_accepts_mixed_raw_and_agent_like(self):
        from langchain_agentkit.extensions import validate_agent_list

        agents = [_make_mock_agent("raw_agent"), _make_agent_like("like_agent")]
        result = validate_agent_list(agents)

        assert "raw_agent" in result
        assert "like_agent" in result

    def test_rejects_duplicate_names(self):
        from langchain_agentkit.extensions import validate_agent_list

        agents = [_make_agent_like("same_name"), _make_agent_like("same_name")]

        with pytest.raises(ValueError, match="Duplicate"):
            validate_agent_list(agents)

    def test_rejects_empty_list(self):
        from langchain_agentkit.extensions import validate_agent_list

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_agent_list([])


# --- AgentExtension with AgentLike ---


class TestAgentExtensionAcceptsAgentLike:
    """AgentExtension should accept AgentLike objects in its agents list."""

    def test_accepts_raw_graphs(self):
        from langchain_agentkit.extensions.agents import AgentExtension

        agents = [_make_mock_agent("researcher", "Research specialist")]
        ext = AgentExtension(agents=agents)

        assert ext._agents_by_name["researcher"] is agents[0]

    def test_accepts_agent_like(self):
        from langchain_agentkit.extensions.agents import AgentExtension

        agent_like = _make_agent_like("researcher", "Research specialist")
        ext = AgentExtension(agents=[agent_like])

        assert "researcher" in ext._agents_by_name

    def test_accepts_mixed(self):
        from langchain_agentkit.extensions.agents import AgentExtension

        raw = _make_mock_agent("raw_agent", "Raw")
        like = _make_agent_like("like_agent", "Like")
        ext = AgentExtension(agents=[raw, like])

        assert "raw_agent" in ext._agents_by_name
        assert "like_agent" in ext._agents_by_name


# --- TeamExtension with AgentLike ---


class TestTeamExtensionAcceptsAgentLike:
    """TeamExtension should accept AgentLike objects in its agents list."""

    def test_accepts_raw_graphs(self):
        from langchain_agentkit.extensions.teams import TeamExtension

        agents = [_make_mock_agent("researcher", "Research")]
        ext = TeamExtension(agents=agents)

        assert "researcher" in ext._agents_by_name

    def test_accepts_agent_like(self):
        from langchain_agentkit.extensions.teams import TeamExtension

        agent_like = _make_agent_like("coder", "Writes code")
        ext = TeamExtension(agents=[agent_like])

        assert "coder" in ext._agents_by_name

    def test_accepts_mixed(self):
        from langchain_agentkit.extensions.teams import TeamExtension

        raw = _make_mock_agent("raw_agent", "Raw")
        like = _make_agent_like("like_agent", "Like")
        ext = TeamExtension(agents=[raw, like])

        assert "raw_agent" in ext._agents_by_name
        assert "like_agent" in ext._agents_by_name
