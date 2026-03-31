"""Tests for SkillsExtension."""

import tempfile
from pathlib import Path

import pytest
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.extensions.skills import SkillsExtension
from langchain_agentkit.extensions.skills.types import SkillConfig

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)

_SKILL_MD = """\
---
name: market-sizing
description: Calculate TAM, SAM, and SOM for market analysis
---
# Market Sizing Methodology

## Step 1: Define Market Boundaries
Identify the total addressable market.
"""


def _make_configs() -> list[SkillConfig]:
    return [
        SkillConfig(
            name="market-sizing",
            description="Calculate TAM, SAM, and SOM",
            prompt="# Market Sizing Methodology",
        ),
    ]


def _write_skill(tmpdir: Path, name: str, content: str) -> None:
    """Write a SKILL.md file into a subdirectory."""
    skill_dir = tmpdir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content)


class TestProgrammaticMode:
    """Mode A: skills passed as list[SkillConfig]."""

    def test_returns_one_tool(self):
        mw = SkillsExtension(skills=_make_configs())

        assert len(mw.tools) == 1

    def test_tool_is_named_skill(self):
        mw = SkillsExtension(skills=_make_configs())

        assert mw.tools[0].name == "Skill"

    def test_skill_tool_returns_instructions(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.tools[0].invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_no_filesystem_needed(self):
        mw = SkillsExtension(skills=_make_configs())

        assert len(mw.tools) == 1

    def test_empty_list_returns_skill_tool(self):
        mw = SkillsExtension(skills=[])

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Skill"

    def test_configs_property_returns_copy(self):
        configs = _make_configs()
        mw = SkillsExtension(skills=configs)

        assert mw.configs == configs
        assert mw.configs is not configs


class TestDirectoryMode:
    """Mode B: skills discovered from directory path."""

    def test_discovers_skills_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)

            mw = SkillsExtension(skills=tmpdir)

            assert len(mw.configs) == 1
            assert mw.configs[0].name == "market-sizing"

    def test_skill_tool_works_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)

            mw = SkillsExtension(skills=tmpdir)
            result = mw.tools[0].invoke({"skill_name": "market-sizing"})

            assert "Market Sizing Methodology" in result

    def test_returns_one_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)

            mw = SkillsExtension(skills=tmpdir)

            assert len(mw.tools) == 1
            assert mw.tools[0].name == "Skill"

    def test_empty_directory_returns_no_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []

    def test_nonexistent_directory_returns_no_configs(self):
        mw = SkillsExtension(skills="/nonexistent/path")

        assert mw.configs == []

    def test_skips_invalid_skill_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "INVALID", "---\nname: INVALID\ndescription: bad\n---\nbody")

            mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []

    def test_skips_skills_without_description(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "test", "---\nname: test\n---\nbody")

            mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []

    def test_deduplicates_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "a", "---\nname: dupe\ndescription: first\n---\nbody1")
            _write_skill(Path(tmpdir), "b", "---\nname: dupe\ndescription: second\n---\nbody2")

            mw = SkillsExtension(skills=tmpdir)

            assert len(mw.configs) == 1

    def test_accepts_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)

            mw = SkillsExtension(skills=Path(tmpdir))

            assert len(mw.configs) == 1


class TestBackendMode:
    """Mode C: skills discovered via BackendProtocol."""

    def test_discovers_skills_from_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backend import OSBackend

            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)

            assert len(mw.configs) == 1
            assert mw.configs[0].name == "market-sizing"

    def test_skill_tool_works_from_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backend import OSBackend

            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)
            result = mw.tools[0].invoke({"skill_name": "market-sizing"})

            assert "Market Sizing Methodology" in result

    def test_empty_backend_returns_no_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backend import OSBackend

            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)

            assert mw.configs == []

    def test_deduplicates_by_name_via_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backend import OSBackend

            _write_skill(Path(tmpdir), "a", "---\nname: dupe\ndescription: first\n---\nbody1")
            _write_skill(Path(tmpdir), "b", "---\nname: dupe\ndescription: second\n---\nbody2")
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)

            assert len(mw.configs) == 1


class TestPrompt:
    def test_returns_string_containing_skills_header(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)
        assert "## Skills" in result

    def test_includes_available_skill_names(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "market-sizing" in result

    def test_includes_progressive_disclosure_instructions(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "progressive disclosure" in result

    def test_no_skills_available_returns_marker(self):
        mw = SkillsExtension(skills=[])

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "(No skills available)" in result


class TestStateSchema:
    def test_state_schema_is_none(self):
        mw = SkillsExtension(skills=_make_configs())

        assert mw.state_schema is None


class TestExtensionProtocol:
    def test_has_tools_property(self):
        assert isinstance(SkillsExtension.tools, property)

    def test_has_prompt_method(self):
        assert callable(getattr(SkillsExtension, "prompt", None))

    def test_has_state_schema_property(self):
        assert isinstance(SkillsExtension.state_schema, property)
