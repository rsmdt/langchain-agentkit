"""Tests for AgentMiddleware."""

from unittest.mock import MagicMock

import pytest

from langchain_agentkit.middleware.agents import AgentMiddleware


def _make_mock_agent(name: str, description: str = "") -> MagicMock:
    """Create a mock agent graph with agentkit metadata."""
    mock = MagicMock()
    mock.agentkit_name = name
    mock.agentkit_description = description
    mock.agentkit_tools_inherit = False
    mock.compile.return_value = MagicMock()
    return mock


class TestAgentMiddlewareConstruction:
    def test_construction_with_valid_agents(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentMiddleware([agent_a, agent_b])

        assert mw._agents_by_name["researcher"] is agent_a
        assert mw._agents_by_name["coder"] is agent_b

    def test_construction_with_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            AgentMiddleware([])

    def test_construction_with_duplicate_names_raises_value_error(self):
        agent_a = _make_mock_agent("researcher")
        agent_b = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="Duplicate agent names"):
            AgentMiddleware([agent_a, agent_b])

    def test_construction_with_missing_agentkit_name_raises_value_error(self):
        mock = MagicMock(spec=[])  # No attributes at all

        with pytest.raises(ValueError, match="agentkit_name"):
            AgentMiddleware([mock])


class TestAgentMiddlewareTools:
    def test_tools_returns_delegate_without_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a], ephemeral=False)

        tool_names = [t.name for t in mw.tools]
        assert "Delegate" in tool_names
        assert "DelegateEphemeral" not in tool_names

    def test_tools_returns_delegate_and_ephemeral_when_enabled(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a], ephemeral=True)

        tool_names = [t.name for t in mw.tools]
        assert "Delegate" in tool_names
        assert "DelegateEphemeral" in tool_names

    def test_tools_returns_immutable_tuple(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentMiddleware([agent_a])

        first = mw.tools
        second = mw.tools

        assert first == second
        assert isinstance(first, tuple)


class TestAgentMiddlewarePrompt:
    def test_prompt_renders_agent_roster(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentMiddleware([agent_a, agent_b])
        result = mw.prompt({})

        assert "researcher" in result
        assert "Research specialist" in result
        assert "coder" in result
        assert "Code specialist" in result

    def test_prompt_includes_delegation_guidelines(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")

        mw = AgentMiddleware([agent_a])
        result = mw.prompt({})

        assert "Delegation Guidelines" in result

    def test_prompt_includes_conciseness_directive_by_default(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a], default_conciseness=True)
        result = mw.prompt({})

        assert "concise" in result.lower()

    def test_prompt_excludes_conciseness_directive_when_disabled(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a], default_conciseness=False)
        result = mw.prompt({})

        # The conciseness directive is the specific appended text
        assert "Synthesize the key findings" not in result

    def test_prompt_shows_no_description_for_undescribed_agents(self):
        agent_a = _make_mock_agent("researcher", "")

        mw = AgentMiddleware([agent_a])
        result = mw.prompt({})

        assert "researcher" in result
        assert "No description" in result

    def test_prompt_returns_string(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a])
        result = mw.prompt({})

        assert isinstance(result, str)


class TestAgentMiddlewareNoStateSchema:
    def test_has_no_state_schema(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentMiddleware([agent_a])

        assert not hasattr(mw, "state_schema")


class TestAgentMiddlewareProtocol:
    def test_satisfies_middleware_protocol(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentMiddleware([agent_a])

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)
        assert isinstance(mw.tools, (list, tuple))
        assert isinstance(mw.prompt({}), str)
