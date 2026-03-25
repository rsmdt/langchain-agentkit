"""Tests for SkillsMiddleware."""

from pathlib import Path

from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.middleware.skills import SkillsMiddleware
from langchain_agentkit.vfs import VirtualFilesystem

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


class TestTools:
    def test_accepts_path_object(self):
        mw = SkillsMiddleware(FIXTURES / "skills")

        assert len(mw.tools) == 6

    def test_returns_six_tools(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        tools = mw.tools

        assert len(tools) == 6

    def test_tool_names(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        names = [t.name for t in mw.tools]

        assert names == ["Skill", "Read", "Write", "Edit", "Glob", "Grep"]

    def test_skill_tool_works(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))
        skill_tool = mw.tools[0]

        result = skill_tool.invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_read_tool_reads_skill_files(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))
        read_tool = mw.tools[1]

        result = read_tool.invoke({"file_path": "/skills/market-sizing/calculator.py"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_read_tool_reads_skill_md(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))
        read_tool = mw.tools[1]

        result = read_tool.invoke({"file_path": "/skills/market-sizing/SKILL.md"})

        assert "market-sizing" in result.lower() or "Market Sizing" in result


class TestFilesystem:
    def test_skills_loaded_into_vfs(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        assert mw.filesystem.exists("/skills/market-sizing/SKILL.md")
        assert mw.filesystem.exists("/skills/market-sizing/calculator.py")

    def test_accepts_external_filesystem(self):
        vfs = VirtualFilesystem()
        vfs.write("/existing/file.txt", "pre-existing")

        mw = SkillsMiddleware(str(FIXTURES / "skills"), filesystem=vfs)

        assert mw.filesystem.exists("/existing/file.txt")
        assert mw.filesystem.exists("/skills/market-sizing/SKILL.md")

    def test_custom_base_path(self):
        mw = SkillsMiddleware(
            str(FIXTURES / "skills"),
            skills_base_path="/custom",
        )

        assert mw.filesystem.exists("/custom/market-sizing/SKILL.md")

    def test_glob_finds_skill_files(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))
        glob_tool = mw.tools[4]

        result = glob_tool.invoke({"pattern": "/skills/*/SKILL.md"})

        assert "/skills/market-sizing/SKILL.md" in result


class TestPrompt:
    def test_returns_string_containing_skills_header(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)
        assert "## Skills" in result

    def test_includes_available_skill_names(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "market-sizing" in result

    def test_includes_progressive_disclosure_instructions(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "progressive disclosure" in result

    def test_includes_read_tool_reference(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "Read(" in result

    def test_no_skills_available_returns_marker(self):
        mw = SkillsMiddleware(str(FIXTURES / "nonexistent_dir"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "(No skills available)" in result

    def test_includes_reference_file_names(self):
        mw = SkillsMiddleware(str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "calculator.py" in result


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
