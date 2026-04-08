"""Tests for FilesystemExtension."""

import tempfile
from pathlib import Path

from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.backends import BackendProtocol, OSBackend
from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.permissions.presets import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
)
from langchain_agentkit.permissions.types import (
    OperationPermissions,
    PermissionRuleset,
)

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


# ---------------------------------------------------------------------------
# Gate 1 — tool removal at registration time
# ---------------------------------------------------------------------------


class TestPermissionGate1:
    """Gate 1 removes tools whose operation is fully denied (no override rules)."""

    def test_readonly_removes_write_edit_bash(self):
        ext = FilesystemExtension(permissions=READONLY_RULESET)
        names = [t.name for t in ext.tools]
        assert "Read" in names
        assert "Glob" in names
        assert "Grep" in names
        assert "Write" not in names
        assert "Edit" not in names
        assert "Bash" not in names

    def test_custom_deny_execute_only(self):
        ruleset = PermissionRuleset(
            default="allow",
            execute=OperationPermissions(default="deny"),
        )
        ext = FilesystemExtension(permissions=ruleset)
        names = [t.name for t in ext.tools]
        assert "Read" in names
        assert "Write" in names
        assert "Bash" not in names

    def test_permissive_keeps_all_tools(self):
        ext = FilesystemExtension(permissions=PERMISSIVE_RULESET)
        names = [t.name for t in ext.tools]
        assert len(names) == 6
        assert "Bash" in names

    def test_default_keeps_tools_with_rules(self):
        """DEFAULT_RULESET has default=ask with rules — Gate 1 should NOT remove."""
        ext = FilesystemExtension(permissions=DEFAULT_RULESET)
        names = [t.name for t in ext.tools]
        # All tools kept because they have rules (secrets/dangerous) even though default=ask
        assert "Write" in names
        assert "Edit" in names
        assert "Bash" in names


# ---------------------------------------------------------------------------
# Gate 2 — per-call permission checking
# ---------------------------------------------------------------------------


class TestPermissionGate2:
    """Gate 2 checks permissions on each tool invocation."""

    def test_permissive_allows_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            write_tool = next(t for t in ext.tools if t.name == "Write")
            result = write_tool.invoke({"file_path": "/test.txt", "content": "hello"})
            # Permission-wrapped tools return (content, artifact) tuple
            content = result[0] if isinstance(result, tuple) else result
            assert "successfully" in content

    def test_permissive_denies_secrets_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            read_tool = next(t for t in ext.tools if t.name == "Read")
            result = read_tool.invoke({"file_path": "/project/.env"})
            assert "Permission denied" in result

    def test_permissive_denies_secrets_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            write_tool = next(t for t in ext.tools if t.name == "Write")
            result = write_tool.invoke({"file_path": "/project/.env", "content": "x"})
            assert "Permission denied" in result

    def test_strict_asks_without_hitl_denies(self):
        """STRICT asks for everything. Without HITL, ask degrades to deny."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=STRICT_RULESET)
            read_tool = next(t for t in ext.tools if t.name == "Read")
            result = read_tool.invoke({"file_path": "/file.txt"})
            assert "Permission required" in result or "approval" in result.lower()


# ---------------------------------------------------------------------------
# Prompt — Bash visibility
# ---------------------------------------------------------------------------


class TestPromptWithPermissions:
    def test_prompt_includes_bash_when_available(self):
        ext = FilesystemExtension(permissions=PERMISSIVE_RULESET)
        result = ext.prompt({}, _TEST_RUNTIME)
        assert "Bash" in result

    def test_prompt_excludes_bash_when_denied(self):
        ext = FilesystemExtension(permissions=READONLY_RULESET)
        result = ext.prompt({}, _TEST_RUNTIME)
        assert "Bash" not in result
