"""Tests for SkillsMiddleware."""

from pathlib import Path

from langchain_core.runnables import RunnableConfig

from langchain_agentkit.skill_kit import SkillKit
from langchain_agentkit.skills_middleware import SkillsMiddleware

FIXTURES = Path(__file__).parent / "fixtures"


class TestTools:
    def test_returns_two_tools(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        tools = mw.tools

        assert len(tools) == 2

    def test_tools_delegates_to_skill_kit(self):
        skills_dir = str(FIXTURES / "skills")
        mw = SkillsMiddleware(skills_dir)
        kit = SkillKit(skills_dir)

        mw_tool_names = [t.name for t in mw.tools]
        kit_tool_names = [t.name for t in kit.tools]

        assert mw_tool_names == kit_tool_names


class TestPrompt:
    def test_returns_string_containing_skills_system(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, RunnableConfig())

        assert isinstance(result, str)
        assert "Skills System" in result

    def test_includes_available_skill_names(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, RunnableConfig())

        assert "market-sizing" in result

    def test_includes_progressive_disclosure_instructions(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, RunnableConfig())

        assert "Progressive Disclosure" in result

    def test_no_skills_available_returns_marker(self):
        mw = SkillsMiddleware(str(FIXTURES / "nonexistent_dir"))

        result = mw.prompt({}, RunnableConfig())

        assert "(No skills available)" in result

    def test_always_returns_string(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, RunnableConfig())

        assert result is not None
        assert isinstance(result, str)


class TestFormatSkillsList:
    def test_includes_skill_name_and_description(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw._format_skills_list()

        assert "market-sizing" in result
        assert "Calculate TAM, SAM, and SOM" in result

    def test_includes_load_instructions(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw._format_skills_list()

        assert 'Skill("market-sizing")' in result

    def test_returns_no_skills_available_for_empty_directory(self):
        mw = SkillsMiddleware(str(FIXTURES / "nonexistent_dir"))

        result = mw._format_skills_list()

        assert result == "(No skills available)"


class TestMiddlewareProtocol:
    def test_has_tools_property(self):
        assert isinstance(
            SkillsMiddleware.tools,
            property,
        )

    def test_has_prompt_method(self):
        assert callable(getattr(SkillsMiddleware, "prompt", None))
