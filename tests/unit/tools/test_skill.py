"""Tests for SkillRegistry."""

from pathlib import Path

from langchain_agentkit.tools.skill import SkillRegistry
from langchain_agentkit.vfs import VirtualFilesystem

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


class TestInit:
    def test_accepts_single_string_path(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        assert len(kit.skills_dirs) == 1

    def test_accepts_path_object(self):
        kit = SkillRegistry(FIXTURES / "skills")

        assert len(kit.skills_dirs) == 1
        assert len(kit.tools) == 1

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
    def test_returns_one_tool(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools

        assert len(tools) == 1

    def test_first_tool_is_skill(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        tools = kit.tools

        assert tools[0].name == "Skill"

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


class TestPopulateFilesystem:
    def test_populates_skill_md(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        vfs = VirtualFilesystem()

        kit.populate_filesystem(vfs)

        assert vfs.exists("/skills/market-sizing/SKILL.md")

    def test_populates_reference_files(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        vfs = VirtualFilesystem()

        kit.populate_filesystem(vfs)

        assert vfs.exists("/skills/market-sizing/calculator.py")

    def test_custom_base_path(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        vfs = VirtualFilesystem()

        kit.populate_filesystem(vfs, base_path="/custom")

        assert vfs.exists("/custom/market-sizing/SKILL.md")

    def test_multiple_directories(self):
        kit = SkillRegistry(
            [
                str(FIXTURES / "skills"),
                str(FIXTURES / "skills_extra"),
            ]
        )
        vfs = VirtualFilesystem()

        kit.populate_filesystem(vfs)

        assert vfs.exists("/skills/market-sizing/SKILL.md")
        assert vfs.exists("/skills/competitive-analysis/SKILL.md")

    def test_skill_md_content_matches_real_file(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))
        vfs = VirtualFilesystem()
        kit.populate_filesystem(vfs)

        real_content = (FIXTURES / "skills" / "market-sizing" / "SKILL.md").read_text()
        virtual_content = vfs.read("/skills/market-sizing/SKILL.md")

        assert virtual_content == real_content


class TestAvailableSkillsDescription:
    def test_includes_reference_files(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        description = kit._build_available_skills_description()

        assert "calculator.py" in description

    def test_includes_skill_name(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        description = kit._build_available_skills_description()

        assert "market-sizing" in description


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
