"""Tests for YAML frontmatter parsing."""

from pathlib import Path

import pytest

from langchain_agentkit.frontmatter import parse_frontmatter

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseFrontmatter:
    def test_parses_name_and_description(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert result.metadata["name"] == "researcher"
        assert result.metadata["description"] == (
            "Research specialist that gathers factual information"
        )

    def test_parses_allowed_tools_as_comma_separated(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/analyst.md")

        assert result.metadata["allowed-tools"] == "sql_query, calculate"

    def test_researcher_has_no_allowed_tools(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert "allowed-tools" not in result.metadata

    def test_extracts_body_as_content(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert "You are a highly capable Research Assistant" in result.content

    def test_body_excludes_frontmatter(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert "---" not in result.content
        assert "name:" not in result.content

    def test_body_contains_skill_references_in_prose(self):
        result = parse_frontmatter(FIXTURES / "prompts/nodes/researcher.md")

        assert 'Skill("market-sizing")' in result.content
        assert "competitive-analysis skill" in result.content

    def test_returns_empty_metadata_when_no_frontmatter(self, tmp_path):
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just plain markdown content.")

        result = parse_frontmatter(md_file)

        assert result.metadata == {}
        assert result.content == "Just plain markdown content."

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_frontmatter(tmp_path / "nonexistent.md")

    def test_parses_skill_frontmatter(self):
        result = parse_frontmatter(FIXTURES / "skills/market-sizing/SKILL.md")

        assert result.metadata["name"] == "market-sizing"
        assert result.metadata["description"] == "Calculate TAM, SAM, and SOM for market analysis"
        assert "# Market Sizing Methodology" in result.content
