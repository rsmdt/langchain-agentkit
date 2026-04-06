"""Tests for FilesystemExtension."""

import tempfile
from pathlib import Path

from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.backend import BackendProtocol, OSBackend
from langchain_agentkit.extensions.filesystem import FilesystemExtension

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestConstructor:
    def test_default_uses_os_backend(self):
        ext = FilesystemExtension()

        assert isinstance(ext.backend, OSBackend)

    def test_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)

            assert isinstance(ext.backend, OSBackend)

    def test_custom_backend(self):

        class StubBackend:
            def read(self, path, offset=0, limit=2000):
                return ""

            def write(self, path, content):
                return {"path": path, "bytes_written": 0}

            def edit(self, path, old_string, new_string, replace_all=False):
                return {"path": path, "replacements": 0}

            def glob(self, pattern, path="/"):
                return []

            def grep(self, pattern, path=None, glob=None, ignore_case=False):
                return []

            def read_bytes(self, path):
                return b""

            def execute(self, command, timeout=None, workdir=None):
                return {"output": "", "exit_code": 0, "truncated": False}

        ext = FilesystemExtension(backend=StubBackend())

        assert isinstance(ext.backend, BackendProtocol)

    def test_root_as_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=Path(tmpdir))

            assert isinstance(ext.backend, OSBackend)


class TestTools:
    def test_returns_six_tools(self):
        ext = FilesystemExtension()

        assert len(ext.tools) == 6

    def test_tool_names(self):
        ext = FilesystemExtension()
        names = [t.name for t in ext.tools]

        assert names == ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]

    def test_read_tool_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file to the temp directory
            (Path(tmpdir) / "hello.txt").write_text("world")

            ext = FilesystemExtension(root=tmpdir)
            read_tool = ext.tools[0]
            result = read_tool.invoke({"file_path": "/hello.txt"})

            assert "world" in result

    def test_write_tool_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            write_tool = ext.tools[1]
            result = write_tool.invoke({"file_path": "/new.txt", "content": "hello"})

            assert "created successfully" in result
            assert (Path(tmpdir) / "new.txt").read_text() == "hello"

    def test_tools_are_cached(self):
        ext = FilesystemExtension()

        assert ext.tools is ext.tools


class TestPrompt:
    def test_returns_filesystem_prompt(self):
        ext = FilesystemExtension()
        result = ext.prompt({}, _TEST_RUNTIME)

        assert "Filesystem" in result
        assert "Read" in result

    def test_shows_root_for_os_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            result = ext.prompt({}, _TEST_RUNTIME)

            assert tmpdir in result


class TestStateSchema:
    def test_returns_none(self):
        ext = FilesystemExtension()

        assert ext.state_schema is None


class TestExtensionProtocol:
    def test_has_tools_property(self):
        assert isinstance(FilesystemExtension.tools, property)

    def test_has_prompt_method(self):
        assert callable(getattr(FilesystemExtension, "prompt", None))

    def test_has_state_schema_property(self):
        assert isinstance(FilesystemExtension.state_schema, property)
