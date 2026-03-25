"""Tests for SkillConfig."""

from pathlib import Path

import pytest

from langchain_agentkit.types import SkillConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestSkillConfig:
    def test_from_directory_parses_name(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert config.name == "market-sizing"

    def test_from_directory_parses_description(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert config.description == "Calculate TAM, SAM, and SOM for market analysis"

    def test_from_directory_parses_instructions(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert "# Market Sizing Methodology" in config.instructions

    def test_from_directory_excludes_frontmatter_from_instructions(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert "---" not in config.instructions
        assert "name:" not in config.instructions

    def test_from_directory_discovers_reference_files(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert "calculator.py" in config.reference_files

    def test_from_directory_excludes_skill_md_from_references(self):
        config = SkillConfig.from_directory(FIXTURES / "skills/market-sizing")

        assert "SKILL.md" not in config.reference_files

    def test_from_directory_sets_directory(self):
        skill_dir = FIXTURES / "skills/market-sizing"
        config = SkillConfig.from_directory(skill_dir)

        assert config.directory == skill_dir

    def test_from_directory_raises_on_missing_skill_md(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SkillConfig.from_directory(tmp_path)

    def test_from_directory_falls_back_to_dir_name_when_no_name(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\ndescription: test\n---\nBody")

        config = SkillConfig.from_directory(skill_dir)

        assert config.name == "my-skill"

    def test_frozen_dataclass(self):
        config = SkillConfig(name="test", description="desc")

        with pytest.raises(AttributeError):
            config.name = "changed"
