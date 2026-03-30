"""Tests for SkillConfig."""

import pytest

from langchain_agentkit.types import SkillConfig


class TestSkillConfig:
    def test_construction_with_required_fields(self):
        config = SkillConfig(name="test", description="desc")

        assert config.name == "test"
        assert config.description == "desc"
        assert config.instructions == ""

    def test_construction_with_instructions(self):
        config = SkillConfig(name="test", description="desc", instructions="# Guide")

        assert config.instructions == "# Guide"

    def test_frozen_dataclass(self):
        config = SkillConfig(name="test", description="desc")

        with pytest.raises(AttributeError):
            config.name = "changed"


class TestFromFrontmatter:
    def test_parses_name_and_description(self):
        metadata = {"name": "market-sizing", "description": "Calculate TAM"}
        content = "# Methodology"

        config = SkillConfig.from_frontmatter(metadata, content)

        assert config.name == "market-sizing"
        assert config.description == "Calculate TAM"
        assert config.instructions == "# Methodology"

    def test_empty_metadata_returns_empty_fields(self):
        config = SkillConfig.from_frontmatter({}, "body")

        assert config.name == ""
        assert config.description == ""
        assert config.instructions == "body"

    def test_missing_keys_default_to_empty(self):
        config = SkillConfig.from_frontmatter({"name": "x"}, "")

        assert config.name == "x"
        assert config.description == ""
        assert config.instructions == ""
