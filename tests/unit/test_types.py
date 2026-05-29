"""Tests for SkillConfig."""

from langchain_agentkit.extensions.skills.types import SkillConfig


class TestFromFrontmatter:
    def test_maps_metadata_and_content_to_config(self):
        config = SkillConfig.from_frontmatter(
            {"name": "market-sizing", "description": "Calculate TAM"},
            "# Methodology",
        )

        assert config.name == "market-sizing"
        assert config.description == "Calculate TAM"
        # content is mapped onto the prompt field
        assert config.prompt == "# Methodology"
