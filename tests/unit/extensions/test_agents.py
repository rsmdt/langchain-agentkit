"""Tests for AgentExtension."""

import tempfile
from pathlib import Path
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


_AGENT_MD = """\
---
name: researcher
description: Research specialist that gathers factual information
---
You are a highly capable Research Assistant.
Use WebSearch to find current information.
"""


class TestProgrammaticMode:
    """Mode A: agents passed as list of executable objects."""

    def test_construction_with_valid_agents(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentExtension(agents=[agent_a, agent_b])

        assert mw._agents_by_name["researcher"] is agent_a
        assert mw._agents_by_name["coder"] is agent_b

    def test_construction_with_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="agents list cannot be empty"):
            AgentExtension(agents=[])

    def test_construction_with_duplicate_names_raises_value_error(self):
        agent_a = _make_mock_agent("researcher")
        agent_b = _make_mock_agent("researcher")

        with pytest.raises(ValueError, match="Duplicate agent names"):
            AgentExtension(agents=[agent_a, agent_b])

    def test_construction_with_missing_name_raises_value_error(self):
        mock = MagicMock(spec=[])  # No attributes at all

        with pytest.raises(ValueError, match="name"):
            AgentExtension(agents=[mock])


class TestDirectoryMode:
    """Mode B: agents discovered from directory path."""

    def test_discovers_agents_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)

            assert "researcher" in mw._agents_by_name

    def test_discovered_agent_has_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)

            agent = mw._agents_by_name["researcher"]
            assert agent.description == "Research specialist that gathers factual information"

    def test_discovered_agent_has_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)

            agent = mw._agents_by_name["researcher"]
            assert "Research Assistant" in agent._agent_config.prompt

    def test_empty_directory_returns_empty_roster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mw = AgentExtension(agents=tmpdir)

            assert mw._agents_by_name == {}

    def test_nonexistent_directory_returns_empty_roster(self):
        mw = AgentExtension(agents="/nonexistent/path")

        assert mw._agents_by_name == {}

    def test_skips_files_without_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "unnamed.md").write_text("---\ndescription: no name\n---\nbody")

            mw = AgentExtension(agents=tmpdir)

            assert mw._agents_by_name == {}

    def test_skips_missing_frontmatter_with_warning(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "plain.md").write_text("No frontmatter, just text.")

            with caplog.at_level(logging.WARNING):
                mw = AgentExtension(agents=tmpdir)

            assert mw._agents_by_name == {}
            assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_skips_malformed_yaml_with_warning(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "bad.md").write_text("---\n: broken: yaml: {{{\n---\nbody")

            with caplog.at_level(logging.WARNING):
                mw = AgentExtension(agents=tmpdir)

            assert mw._agents_by_name == {}
            assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_broken_frontmatter_excluded_valid_kept(self, caplog):
        """Mixed directory: broken files excluded, valid files loaded."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid agent
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)
            # No frontmatter
            (Path(tmpdir) / "plain.md").write_text("Just text, no delimiters.")
            # Malformed YAML
            (Path(tmpdir) / "broken.md").write_text("---\n: bad: {{{\n---\nbody")
            # Missing name
            (Path(tmpdir) / "noname.md").write_text("---\ndescription: orphan\n---\nbody")
            # Invalid name format
            (Path(tmpdir) / "upper.md").write_text("---\nname: UPPER\ndescription: x\n---\nb")

            with caplog.at_level(logging.WARNING):
                mw = AgentExtension(agents=tmpdir)

            assert list(mw._agents_by_name.keys()) == ["researcher"]

            skip_warnings = [r for r in caplog.records if "skipping" in r.message.lower()]
            assert len(skip_warnings) == 4

    def test_deduplicates_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.md").write_text("---\nname: dupe\ndescription: first\n---\nbody1")
            (Path(tmpdir) / "b.md").write_text("---\nname: dupe\ndescription: second\n---\nbody2")

            mw = AgentExtension(agents=tmpdir)

            assert len(mw._agents_by_name) == 1

    def test_marks_filesystem_agents_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)

            assert mw._has_config_agents is True

    def test_programmatic_mode_has_no_filesystem_flag(self):
        agent = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent])

        assert mw._has_config_agents is False


class TestBackendMode:
    """Mode C: agents discovered via BackendProtocol (async setup)."""

    async def test_discovers_agents_from_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)
            backend = OSBackend(tmpdir)

            mw = AgentExtension(agents="/", backend=backend)
            await mw.setup(extensions=[mw])

            assert "researcher" in mw._agents_by_name

    async def test_discovered_agent_has_description_via_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)
            backend = OSBackend(tmpdir)

            mw = AgentExtension(agents="/", backend=backend)
            await mw.setup(extensions=[mw])

            agent = mw._agents_by_name["researcher"]
            assert agent.description == "Research specialist that gathers factual information"

    async def test_empty_backend_returns_empty_roster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            backend = OSBackend(tmpdir)

            mw = AgentExtension(agents="/", backend=backend)
            await mw.setup(extensions=[mw])

            assert mw._agents_by_name == {}

    async def test_deduplicates_by_name_via_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            (Path(tmpdir) / "a.md").write_text("---\nname: dupe\ndescription: first\n---\nbody1")
            (Path(tmpdir) / "b.md").write_text("---\nname: dupe\ndescription: second\n---\nbody2")
            backend = OSBackend(tmpdir)

            mw = AgentExtension(agents="/", backend=backend)
            await mw.setup(extensions=[mw])

            assert len(mw._agents_by_name) == 1


class TestTools:
    def test_tools_returns_single_agent_tool(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], ephemeral=False)

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Agent"

    def test_tools_returns_single_agent_tool_with_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], ephemeral=True)

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Agent"

    def test_tools_returns_immutable_tuple(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentExtension(agents=[agent_a])

        first = mw.tools
        second = mw.tools

        assert first == second
        assert isinstance(first, tuple)

    def test_directory_agents_provide_agent_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)

            assert len(mw.tools) == 1
            assert mw.tools[0].name == "Agent"


class TestPrompt:
    def test_prompt_renders_agent_roster(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")
        agent_b = _make_mock_agent("coder", "Code specialist")

        mw = AgentExtension(agents=[agent_a, agent_b])
        result = mw.prompt({})

        assert "researcher" in result
        assert "Research specialist" in result
        assert "coder" in result
        assert "Code specialist" in result

    def test_prompt_includes_delegation_guidelines(self):
        agent_a = _make_mock_agent("researcher", "Research specialist")

        mw = AgentExtension(agents=[agent_a])
        result = mw.prompt({})

        assert "Delegation Guidelines" in result

    def test_prompt_references_agent_tool(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a])
        result = mw.prompt({})

        assert "Agent" in result

    def test_prompt_includes_conciseness_directive_by_default(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], default_conciseness=True)
        result = mw.prompt({})

        assert "concise" in result.lower()

    def test_prompt_excludes_conciseness_directive_when_disabled(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], default_conciseness=False)
        result = mw.prompt({})

        assert "Synthesize the key findings" not in result

    def test_prompt_shows_no_description_for_undescribed_agents(self):
        agent_a = _make_mock_agent("researcher", "")

        mw = AgentExtension(agents=[agent_a])
        result = mw.prompt({})

        assert "researcher" in result
        assert "No description" in result

    def test_prompt_includes_dynamic_section_when_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], ephemeral=True)
        result = mw.prompt({})

        assert "custom agent" in result.lower()
        assert "prompt" in result

    def test_prompt_excludes_dynamic_section_when_not_ephemeral(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a], ephemeral=False)
        result = mw.prompt({})

        assert "custom agent" not in result.lower()

    def test_prompt_returns_string(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a])
        result = mw.prompt({})

        assert isinstance(result, str)

    def test_directory_agents_appear_in_roster(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "researcher.md").write_text(_AGENT_MD)

            mw = AgentExtension(agents=tmpdir)
            result = mw.prompt({})

            assert "researcher" in result
            assert "Research specialist" in result


class TestNoStateSchema:
    def test_state_schema_is_none(self):
        agent_a = _make_mock_agent("researcher")

        mw = AgentExtension(agents=[agent_a])

        assert mw.state_schema is None


class TestGetToolsDescription:
    """Tests for _get_tools_description agent roster helper."""

    def test_no_config_returns_star(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description

        agent = _make_mock_agent("plain")
        del agent._agent_config

        result = _get_tools_description(agent)

        assert result == "*"

    def test_no_restrictions_returns_all_tools(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description
        from langchain_agentkit.extensions.agents.types import AgentConfig

        agent = _make_mock_agent("open")
        agent._agent_config = AgentConfig(name="open", description="", prompt="test")

        result = _get_tools_description(agent)

        assert result == "All tools"

    def test_denylist_only(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description

        agent = _make_mock_agent("restricted")
        config = MagicMock()
        config.tools = None
        config.disallowed_tools = ["Edit"]
        agent._agent_config = config

        result = _get_tools_description(agent)

        assert result == "All tools except Edit"

    def test_allowlist_only(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description
        from langchain_agentkit.extensions.agents.types import AgentConfig

        agent = _make_mock_agent("limited")
        agent._agent_config = AgentConfig(
            name="limited", description="", prompt="test", tools=["Read", "Grep"]
        )

        result = _get_tools_description(agent)

        assert result == "Grep, Read"

    def test_both_lists_effective_set(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description

        agent = _make_mock_agent("mixed")
        config = MagicMock()
        config.tools = ["Read", "Edit"]
        config.disallowed_tools = ["Edit"]
        agent._agent_config = config

        result = _get_tools_description(agent)

        assert result == "Read"

    def test_empty_effective_set(self):
        from langchain_agentkit.extensions.agents.extension import _get_tools_description

        agent = _make_mock_agent("none")
        config = MagicMock()
        config.tools = ["Edit"]
        config.disallowed_tools = ["Edit"]
        agent._agent_config = config

        result = _get_tools_description(agent)

        assert result == "No tools"


class TestExtensionProtocol:
    def test_satisfies_extension_protocol(self):
        agent_a = _make_mock_agent("researcher")
        mw = AgentExtension(agents=[agent_a])

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)
        assert isinstance(mw.tools, (list, tuple))
        assert isinstance(mw.prompt({}), str)
