"""Tests for SkillRegistry."""

from pathlib import Path

from langchain_agentkit.skill_registry import SkillRegistry

FIXTURES = Path(__file__).parent / "fixtures"


class TestInit:
    def test_accepts_single_string_path(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        assert len(kit.skills_dirs) == 1

    def test_accepts_path_object(self):
        kit = SkillRegistry(FIXTURES / "skills")

        assert len(kit.skills_dirs) == 1
        assert len(kit.tools) == 2

    def test_accepts_list_of_paths(self):
        kit = SkillRegistry(
            [
                str(FIXTURES / "skills"),
                str(FIXTURES / "skills_extra"),
            ]
        )

        assert len(kit.skills_dirs) == 2

    def test_accepts_list_of_path_objects(self):
        kit = SkillRegistry(
            [
                FIXTURES / "skills",
                FIXTURES / "skills_extra",
            ]
        )

        assert len(kit.skills_dirs) == 2
        assert "market-sizing" in kit.tools[0].description


class TestTools:
    def test_returns_two_tools(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools

        assert len(tools) == 2

    def test_first_tool_is_skill(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools

        assert tools[0].name == "Skill"

    def test_second_tool_is_skill_read(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools

        assert tools[1].name == "SkillRead"

    def test_skill_description_lists_available_skills(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools
        skill_tool = tools[0]

        assert "market-sizing" in skill_tool.description

    def test_tools_property_is_cached(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        first = kit.tools
        second = kit.tools

        assert first is second


class TestSkillTool:
    def test_loads_skill_instructions(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_tool = kit.tools[0]

        result = skill_tool.invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_unknown_skill_returns_error(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_tool = kit.tools[0]

        result = skill_tool.invoke({"skill_name": "nonexistent"})

        assert "not found" in result

    def test_invalid_skill_name_returns_error(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_tool = kit.tools[0]

        result = skill_tool.invoke({"skill_name": "../escape"})

        assert "Invalid skill name" in result


class TestSkillReadTool:
    def test_reads_reference_file(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_read_tool = kit.tools[1]

        result = skill_read_tool.invoke(
            {
                "skill_name": "market-sizing",
                "file_name": "calculator.py",
            }
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_file_returns_error(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_read_tool = kit.tools[1]

        result = skill_read_tool.invoke(
            {
                "skill_name": "market-sizing",
                "file_name": "nonexistent.py",
            }
        )

        assert "not found" in result

    def test_path_traversal_returns_error(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        skill_read_tool = kit.tools[1]

        result = skill_read_tool.invoke(
            {
                "skill_name": "market-sizing",
                "file_name": "../../etc/passwd",
            }
        )

        assert "Invalid file name" in result


class TestMultipleDirectories:
    def test_discovers_skills_from_both_directories(self):
        kit = SkillRegistry(
            [
                str(FIXTURES / "skills"),
                str(FIXTURES / "skills_extra"),
            ]
        )

        tools = kit.tools
        skill_tool = tools[0]

        assert "market-sizing" in skill_tool.description
        assert "competitive-analysis" in skill_tool.description

    def test_loads_skill_from_extra_directory(self):
        kit = SkillRegistry(
            [
                str(FIXTURES / "skills"),
                str(FIXTURES / "skills_extra"),
            ]
        )
        skill_tool = kit.tools[0]

        result = skill_tool.invoke({"skill_name": "competitive-analysis"})

        assert "Competitive Analysis" in result
