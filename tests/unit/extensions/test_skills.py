"""Tests for SkillsExtension."""

import tempfile
from pathlib import Path

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

    def test_skips_missing_frontmatter_with_warning(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "plain", "No frontmatter here, just markdown.")

            with caplog.at_level(logging.WARNING):
                mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []
            assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_skips_malformed_yaml_with_warning(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "bad", "---\n: broken: yaml: {{{\n---\nbody")

            with caplog.at_level(logging.WARNING):
                mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []
            assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_logs_warning_for_invalid_name(self, caplog):
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_skill(Path(tmpdir), "BAD", "---\nname: BAD\ndescription: bad\n---\nbody")

            with caplog.at_level(logging.WARNING):
                mw = SkillsExtension(skills=tmpdir)

            assert mw.configs == []
            assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_broken_frontmatter_excluded_valid_kept(self, caplog):
        """Mixed directory: broken files excluded, valid files loaded."""
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid skill
            _write_skill(Path(tmpdir), "good", _SKILL_MD)
            # No frontmatter
            _write_skill(Path(tmpdir), "plain", "Just markdown, no delimiters.")
            # Malformed YAML
            _write_skill(Path(tmpdir), "broken", "---\n: bad: {{{\n---\nbody")
            # Missing required name
            _write_skill(Path(tmpdir), "noname", "---\ndescription: orphan\n---\nbody")
            # Invalid name format
            _write_skill(Path(tmpdir), "UPPER", "---\nname: UPPER\ndescription: bad\n---\nb")

            with caplog.at_level(logging.WARNING):
                mw = SkillsExtension(skills=tmpdir)

            # Only the valid skill loaded
            assert len(mw.configs) == 1
            assert mw.configs[0].name == "market-sizing"

            # Four broken files logged
            skip_warnings = [r for r in caplog.records if "skipping" in r.message.lower()]
            assert len(skip_warnings) == 4

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
    """Mode C: skills discovered via BackendProtocol (async setup)."""

    async def test_discovers_skills_from_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)
            await mw.setup()

            assert len(mw.configs) == 1
            assert mw.configs[0].name == "market-sizing"

    async def test_skill_tool_works_from_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            _write_skill(Path(tmpdir), "market-sizing", _SKILL_MD)
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)
            await mw.setup()
            result = mw.tools[0].invoke({"skill_name": "market-sizing"})

            assert "Market Sizing Methodology" in result

    async def test_empty_backend_returns_no_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)
            await mw.setup()

            assert mw.configs == []

    async def test_deduplicates_by_name_via_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from langchain_agentkit.backends import OSBackend

            _write_skill(Path(tmpdir), "a", "---\nname: dupe\ndescription: first\n---\nbody1")
            _write_skill(Path(tmpdir), "b", "---\nname: dupe\ndescription: second\n---\nbody2")
            backend = OSBackend(tmpdir)

            mw = SkillsExtension(skills="/", backend=backend)
            await mw.setup()

            assert len(mw.configs) == 1


class TestPrompt:
    def test_returns_dict_with_prompt_and_reminder(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, dict)
        assert "## Skills" in result["prompt"]
        assert "reminder" in result
        assert "- market-sizing:" in result["reminder"]

    def test_includes_available_skill_names(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "market-sizing" in result["prompt"]

    def test_includes_progressive_disclosure_instructions(self):
        mw = SkillsExtension(skills=_make_configs())

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "progressive disclosure" in result["prompt"]

    def test_no_skills_available_returns_marker(self):
        mw = SkillsExtension(skills=[])

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "(No skills available)" in result["prompt"]
        # No reminder section when no skills configured.
        assert "reminder" not in result or not result.get("reminder")

    def test_prompt_cache_scope_is_static(self):
        assert SkillsExtension.prompt_cache_scope == "static"


class TestStateSchema:
    def test_state_schema_is_none(self):
        mw = SkillsExtension(skills=_make_configs())

        assert mw.state_schema is None


class TestSkillBudget:
    def test_default_no_budget_no_truncation(self):
        configs = [
            SkillConfig(
                name="long-skill",
                description="A very long description that should not be truncated",
                prompt="body",
            ),
        ]
        mw = SkillsExtension(skills=configs)

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "A very long description that should not be truncated" in result["prompt"]

    def test_max_description_chars_truncates(self):
        configs = [
            SkillConfig(
                name="truncated",
                description="This is a very long description that exceeds the limit",
                prompt="body",
            ),
        ]
        mw = SkillsExtension(skills=configs, max_description_chars=20)

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "This is a very long " in result["prompt"]
        assert "exceeds the limit" not in result["prompt"]

    def test_budget_percent_limits_listing(self):
        configs = [
            SkillConfig(name=f"skill-{i}", description=f"Description {i}" * 20, prompt="body")
            for i in range(50)
        ]
        mw = SkillsExtension(skills=configs, budget_percent=0.0001, context_window=1000)

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "... and" in result["prompt"]
        assert "more skills" in result["prompt"]

    def test_budget_params_default_to_none(self):
        mw = SkillsExtension(skills=_make_configs())

        assert mw._budget_percent is None
        assert mw._max_description_chars is None
        assert mw._context_window is None


class TestExtensionProtocol:
    def test_has_tools_property(self):
        assert isinstance(SkillsExtension.tools, property)

    def test_has_prompt_method(self):
        assert callable(getattr(SkillsExtension, "prompt", None))

    def test_has_state_schema_property(self):
        assert isinstance(SkillsExtension.state_schema, property)
