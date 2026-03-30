"""Tests for build_skill_tool."""

from langchain_agentkit.tools.skill import build_skill_tool
from langchain_agentkit.types import SkillConfig


def _make_configs() -> list[SkillConfig]:
    return [
        SkillConfig(
            name="market-sizing",
            description="Calculate TAM, SAM, and SOM for market analysis",
            prompt="# Market Sizing Methodology\n\nStep 1: Define boundaries.",
        ),
        SkillConfig(
            name="research",
            description="Web research methodology",
            prompt="# Research Guide\n\nUse multiple sources.",
        ),
    ]


class TestBuildSkillTool:
    def test_returns_tool_named_skill(self):
        tool = build_skill_tool(_make_configs())

        assert tool.name == "Skill"

    def test_description_lists_available_skills(self):
        tool = build_skill_tool(_make_configs())

        assert "market-sizing" in tool.description
        assert "research" in tool.description

    def test_empty_configs_returns_tool(self):
        tool = build_skill_tool([])

        assert tool.name == "Skill"


class TestSkillToolInvocation:
    def test_loads_skill_instructions(self):
        tool = build_skill_tool(_make_configs())

        result = tool.invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_unknown_skill_returns_error(self):
        tool = build_skill_tool(_make_configs())

        result = tool.invoke({"skill_name": "nonexistent"})

        assert "not found" in result

    def test_invalid_skill_name_returns_error(self):
        tool = build_skill_tool(_make_configs())

        result = tool.invoke({"skill_name": "../escape"})

        assert "Invalid skill name" in result

    def test_lists_available_skills_in_error(self):
        tool = build_skill_tool(_make_configs())

        result = tool.invoke({"skill_name": "nonexistent"})

        assert "market-sizing" in result
        assert "research" in result
