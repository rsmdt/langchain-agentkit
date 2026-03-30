"""Tests for AgentExtension."""

from unittest.mock import MagicMock

import pytest

from langchain_agentkit.extensions.agents import AgentExtension


def _make_mock_agent(name: str, description: str = "") -> MagicMock:
    """Create a mock agent graph with agentkit metadata."""
    mock = MagicMock()
    mock.name = name
    mock.description = description
    mock.tools_inherit = False
    mock.compile.return_value = MagicMock()
    return mock


class TestAgentExtensionConstruction:
    def test_construction_with_valid_agents(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentExtension([agent_a, agent_b])

        assert mw._agents_by_name["researcher"] is agent_a
        assert mw._agents_by_name["coder"] is agent_b

    def test_construction_with_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            AgentExtension([])

    def test_construction_with_duplicate_names_raises_value_error(self):
        agent_a = _make_mock_agent("researcher")
        agent_b = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="Duplicate agent names"):
            AgentExtension([agent_a, agent_b])

    def test_construction_with_missing_name_raises_value_error(self):
        mock = MagicMock(spec=[])  # No attributes at all

        with pytest.raises(ValueError, match="name"):
            AgentExtension([mock])


class TestAgentExtensionTools:
    def test_tools_returns_single_agent_tool(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], ephemeral=False)

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Agent"

    def test_tools_returns_single_agent_tool_with_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], ephemeral=True)

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Agent"

    def test_tools_returns_immutable_tuple(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentExtension([agent_a])

        first = mw.tools
        second = mw.tools

        assert first == second
        assert isinstance(first, tuple)


class TestAgentExtensionPrompt:
    def test_prompt_renders_agent_roster(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentExtension([agent_a, agent_b])
        result = mw.prompt({})

        assert "researcher" in result
        assert "Research specialist" in result
        assert "coder" in result
        assert "Code specialist" in result

    def test_prompt_includes_delegation_guidelines(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")

        mw = AgentExtension([agent_a])
        result = mw.prompt({})

        assert "Delegation Guidelines" in result

    def test_prompt_references_agent_tool(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a])
        result = mw.prompt({})

        assert "Agent" in result

    def test_prompt_includes_conciseness_directive_by_default(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], default_conciseness=True)
        result = mw.prompt({})

        assert "concise" in result.lower()

    def test_prompt_excludes_conciseness_directive_when_disabled(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], default_conciseness=False)
        result = mw.prompt({})

        assert "Synthesize the key findings" not in result

    def test_prompt_shows_no_description_for_undescribed_agents(self):
        agent_a = _make_mock_agent("researcher", "")

        mw = AgentExtension([agent_a])
        result = mw.prompt({})

        assert "researcher" in result
        assert "No description" in result

    def test_prompt_includes_dynamic_section_when_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], ephemeral=True)
        result = mw.prompt({})

        assert "custom agent" in result.lower()
        assert "prompt" in result

    def test_prompt_excludes_dynamic_section_when_not_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a], ephemeral=False)
        result = mw.prompt({})

        assert "custom agent" not in result.lower()

    def test_prompt_returns_string(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a])
        result = mw.prompt({})

        assert isinstance(result, str)


class TestAgentExtensionNoStateSchema:
    def test_state_schema_is_none(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension([agent_a])

        assert mw.state_schema is None


class TestAgentExtensionProtocol:
    def test_satisfies_extension_protocol(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentExtension([agent_a])

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)
        assert isinstance(mw.tools, (list, tuple))
        assert isinstance(mw.prompt({}), str)
