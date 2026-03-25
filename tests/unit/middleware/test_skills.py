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


class TestToolsWithOwnedFilesystem:
    """When no filesystem is passed, SkillsMiddleware owns VFS and bundles file tools."""

    def test_returns_six_tools(self):
        mw = SkillsMiddleware(skills=FIXTURES / "skills")

        assert len(mw.tools) == 6

    def test_tool_names(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        names = [t.name for t in mw.tools]

        assert names == ["Skill", "Read", "Write", "Edit", "Glob", "Grep"]

    def test_skill_tool_works(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))
        skill_tool = mw.tools[0]

        result = skill_tool.invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_read_tool_reads_skill_files(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))
        read_tool = mw.tools[1]

        result = read_tool.invoke({"file_path": "/skills/market-sizing/calculator.py"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_skills_loaded_into_vfs(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        assert mw.filesystem.exists("/skills/market-sizing/SKILL.md")
        assert mw.filesystem.exists("/skills/market-sizing/calculator.py")

    def test_glob_finds_skill_files(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))
        glob_tool = mw.tools[4]

        result = glob_tool.invoke({"pattern": "/skills/*/SKILL.md"})

        assert "/skills/market-sizing/SKILL.md" in result


class TestToolsWithExternalFilesystem:
    """When filesystem is passed, SkillsMiddleware provides only Skill tool."""

    def test_returns_one_tool(self):
        vfs = VirtualFilesystem()
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"), filesystem=vfs)

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "Skill"

    def test_populates_external_vfs(self):
        vfs = VirtualFilesystem()
        SkillsMiddleware(skills=str(FIXTURES / "skills"), filesystem=vfs)

        assert vfs.exists("/skills/market-sizing/SKILL.md")
        assert vfs.exists("/skills/market-sizing/calculator.py")

    def test_preserves_existing_files_in_vfs(self):
        vfs = VirtualFilesystem()
        vfs.write("/existing/file.txt", "pre-existing")

        SkillsMiddleware(skills=str(FIXTURES / "skills"), filesystem=vfs)

        assert vfs.exists("/existing/file.txt")
        assert vfs.exists("/skills/market-sizing/SKILL.md")

    def test_skill_tool_works(self):
        vfs = VirtualFilesystem()
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"), filesystem=vfs)

        result = mw.tools[0].invoke({"skill_name": "market-sizing"})

        assert "# Market Sizing Methodology" in result

    def test_custom_base_path(self):
        vfs = VirtualFilesystem()
        SkillsMiddleware(
            skills=str(FIXTURES / "skills"),
            filesystem=vfs,
            skills_base_path="/custom",
        )

        assert vfs.exists("/custom/market-sizing/SKILL.md")


class TestPrompt:
    def test_returns_string_containing_skills_header(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)
        assert "## Skills" in result

    def test_includes_available_skill_names(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "market-sizing" in result

    def test_includes_progressive_disclosure_instructions(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "progressive disclosure" in result

    def test_includes_read_tool_reference(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "Read(" in result

    def test_no_skills_available_returns_marker(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "nonexistent_dir"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "(No skills available)" in result

    def test_includes_reference_file_names(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "calculator.py" in result


class TestStateSchema:
    def test_state_schema_is_none(self):
        mw = SkillsMiddleware(skills=str(FIXTURES / "skills"))

        assert mw.state_schema is None


class TestMiddlewareProtocol:
    def test_has_tools_property(self):
        assert isinstance(SkillsMiddleware.tools, property)

    def test_has_prompt_method(self):
        assert callable(getattr(SkillsMiddleware, "prompt", None))

    def test_has_state_schema_property(self):
        assert isinstance(SkillsMiddleware.state_schema, property)
