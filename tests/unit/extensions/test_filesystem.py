"""Tests for FilesystemExtension."""

from pathlib import Path

import pytest
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.backend import MemoryBackend
from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.vfs import VirtualFilesystem

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestConstructor:
    def test_empty(self):
        mw = FilesystemExtension()

        assert len(mw.filesystem) == 0

    def test_with_vfs(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "hello")

        mw = FilesystemExtension(filesystem=vfs)

        assert mw.filesystem is vfs
        assert mw.filesystem.exists("/a.txt")

    def test_with_files_dict(self):
        mw = FilesystemExtension(
            files={
                "/config.json": '{"key": "value"}',
                "/data.txt": "data",
            }
        )

        assert mw.filesystem.exists("/config.json")
        assert mw.filesystem.exists("/data.txt")
        assert mw.filesystem.read("/data.txt") == "data"

    def test_with_files_directory(self):
        mw = FilesystemExtension(files=FIXTURES / "skills" / "market-sizing")

        assert mw.filesystem.exists("/SKILL.md")
        assert mw.filesystem.exists("/calculator.py")

    def test_with_files_string_path(self):
        mw = FilesystemExtension(
            files=str(FIXTURES / "skills" / "market-sizing"),
        )

        assert mw.filesystem.exists("/SKILL.md")

    def test_filesystem_and_files_mutually_exclusive(self):
        vfs = VirtualFilesystem()

        with pytest.raises(ValueError, match="Cannot pass both"):
            FilesystemExtension(filesystem=vfs, files={"/a": "b"})


class TestTools:
    def test_returns_seven_tools(self):
        ext = FilesystemExtension()

        assert len(ext.tools) == 7

    def test_tool_names(self):
        ext = FilesystemExtension()
        names = [t.name for t in ext.tools]

        assert names == ["Read", "Write", "Edit", "Glob", "Grep", "LS", "MultiEdit"]

    def test_execute_tool_included_with_sandbox(self):
        from langchain_agentkit.backend import SandboxProtocol

        class MockSandbox(MemoryBackend):
            def execute(self, command, timeout=None, workdir=None):
                return {"output": "", "exit_code": 0, "truncated": False}

        ext = FilesystemExtension(backend=MockSandbox(), include_execute=True)
        names = [t.name for t in ext.tools]
        assert "Execute" in names

    def test_execute_tool_excluded_by_default(self):
        ext = FilesystemExtension()
        names = [t.name for t in ext.tools]
        assert "Execute" not in names

    def test_read_tool_works_with_preloaded_dict(self):
        mw = FilesystemExtension(files={"/hello.txt": "world"})
        read_tool = mw.tools[0]

        result = read_tool.invoke({"file_path": "/hello.txt"})

        assert "world" in result

    def test_read_tool_works_with_preloaded_directory(self):
        mw = FilesystemExtension(files=FIXTURES / "skills" / "market-sizing")
        read_tool = mw.tools[0]

        result = read_tool.invoke({"file_path": "/SKILL.md"})

        assert "market-sizing" in result.lower() or "Market" in result


class TestPrompt:
    def test_empty_filesystem_returns_none(self):
        mw = FilesystemExtension()

        assert mw.prompt({}, _TEST_RUNTIME) is None

    def test_populated_returns_prompt(self):
        mw = FilesystemExtension(files={"/data/file.txt": "content"})

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "Virtual Filesystem" in result
        assert "1 file(s)" in result
        assert "/data/" in result

    def test_multiple_dirs_listed(self):
        mw = FilesystemExtension(
            files={
                "/data/a.txt": "",
                "/config/b.txt": "",
            }
        )

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "/config/" in result
        assert "/data/" in result


class TestStateSchema:
    def test_returns_none(self):
        mw = FilesystemExtension()

        assert mw.state_schema is None


class TestExtensionProtocol:
    def test_has_tools_property(self):
        assert isinstance(FilesystemExtension.tools, property)

    def test_has_prompt_method(self):
        assert callable(getattr(FilesystemExtension, "prompt", None))

    def test_has_state_schema_property(self):
        assert isinstance(FilesystemExtension.state_schema, property)
