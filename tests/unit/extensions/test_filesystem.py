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

    async def test_read_tool_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a file to the temp directory
            (Path(tmpdir) / "hello.txt").write_text("world")

            ext = FilesystemExtension(root=tmpdir)
            read_tool = ext.tools[0]
            result = await read_tool.ainvoke({"file_path": "/hello.txt"})

            assert "world" in result

    async def test_write_tool_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            write_tool = ext.tools[1]
            result = await write_tool.ainvoke({"file_path": "/new.txt", "content": "hello"})

            assert "created successfully" in result
            assert (Path(tmpdir) / "new.txt").read_text() == "hello"

    def test_tools_are_cached(self):
        ext = FilesystemExtension()

        assert ext.tools is ext.tools


class TestPrompt:
    def test_returns_none_before_setup(self):
        """Without setup(), env is not probed and prompt returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            assert ext.prompt({}, _TEST_RUNTIME) is None

    async def test_returns_env_block_after_setup(self):
        """setup() probes the backend; prompt renders an <env> block."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            await ext.setup(extensions=[])
            result = ext.prompt({}, _TEST_RUNTIME)

            assert result is not None
            assert result.startswith("<env>\n")
            assert result.endswith("\n</env>")
            # Required fields render as labeled lines.
            assert "Working directory: " in result
            assert "OS: " in result
            assert "Shell: " in result
            assert "Available: " in result
            # Must not mention tool names.
            for name in ("Read", "Write", "Edit", "Glob", "Grep", "Bash"):
                assert name not in result


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

    async def test_permissive_allows_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            write_tool = next(t for t in ext.tools if t.name == "Write")
            result = await write_tool.ainvoke({"file_path": "/test.txt", "content": "hello"})
            assert "successfully" in result

    async def test_permissive_denies_secrets_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            read_tool = next(t for t in ext.tools if t.name == "Read")
            result = await read_tool.ainvoke({"file_path": "/project/.env"})
            assert "Permission denied" in result

    async def test_permissive_denies_secrets_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            write_tool = next(t for t in ext.tools if t.name == "Write")
            result = await write_tool.ainvoke({"file_path": "/project/.env", "content": "x"})
            assert "Permission denied" in result

    async def test_strict_asks_without_hitl_denies(self):
        """STRICT asks for everything. Without HITL, ask degrades to deny."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=STRICT_RULESET)
            read_tool = next(t for t in ext.tools if t.name == "Read")
            result = await read_tool.ainvoke({"file_path": "/file.txt"})
            assert "Permission required" in result or "approval" in result.lower()

    async def test_permissive_denies_write_under_agentkit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            write_tool = next(t for t in ext.tools if t.name == "Write")
            result = await write_tool.ainvoke(
                {"file_path": "/workspace/.agentkit/skills/evil.md", "content": "x"}
            )
            assert "Permission denied" in result

    async def test_permissive_denies_edit_under_agentkit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            edit_tool = next(t for t in ext.tools if t.name == "Edit")
            result = await edit_tool.ainvoke(
                {
                    "file_path": "/workspace/.agentkit/AGENTS.md",
                    "old_string": "a",
                    "new_string": "b",
                }
            )
            assert "Permission denied" in result

    async def test_permissive_denies_bash_agentkit_modification(self):
        """Full enforcement path: Bash wrapped via FilesystemExtension must
        deny commands referencing .agentkit before they reach the backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            bash_tool = next(t for t in ext.tools if t.name == "Bash")
            result = await bash_tool.ainvoke({"command": "echo pwn > .agentkit/skills/evil.md"})
            assert "Permission denied" in result

    async def test_permissive_denies_bash_agentkit_read_via_shell(self):
        """Coarse by design: even shell reads of .agentkit are denied under
        the current glob. Documents the intentional over-denial."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            bash_tool = next(t for t in ext.tools if t.name == "Bash")
            result = await bash_tool.ainvoke({"command": "cat .agentkit/AGENTS.md"})
            assert "Permission denied" in result

    async def test_permissive_allows_benign_bash(self):
        """Regression guard: the new execute denies are targeted — benign
        commands must still pass through PERMISSIVE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            bash_tool = next(t for t in ext.tools if t.name == "Bash")
            result = await bash_tool.ainvoke({"command": "echo hello"})
            # Bash returns (content, artifact) — stdout appears in either form.
            text = result if isinstance(result, str) else str(result)
            assert "Permission denied" not in text


# ---------------------------------------------------------------------------
# Gate 2 — missing target argument defense-in-depth
# ---------------------------------------------------------------------------


class TestPermissionWrapperMissingTarget:
    """``_wrap_with_permission_check`` must refuse to run its underlying
    coroutine when the configured target_arg is not present in kwargs —
    otherwise a malformed tool call would check permissions against the
    literal string ``"*"`` and silently fall through to the operation
    default, which is a defense-in-depth gap under PERMISSIVE."""

    async def test_missing_target_arg_raises_tool_exception(self):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        from langchain_agentkit.extensions.filesystem.extension import (
            _wrap_with_permission_check,
        )

        class _Schema(BaseModel):
            foo: str
            bar: str

        async def _impl(foo: str, bar: str) -> str:
            return f"{foo}+{bar}"

        inner = StructuredTool.from_function(
            coroutine=_impl,
            name="StubTool",
            description="stub",
            args_schema=_Schema,
            handle_tool_error=True,
        )

        wrapped = _wrap_with_permission_check(
            tool=inner,
            operation="write",
            target_arg="nonexistent_arg",
            permissions=PERMISSIVE_RULESET,
            hitl_check=lambda: False,
        )

        # handle_tool_error=True converts ToolException into a string result.
        result = await wrapped.ainvoke({"foo": "x", "bar": "y"})

        assert "without a target argument" in result
        assert "nonexistent_arg" in result

    async def test_empty_target_arg_raises_tool_exception(self):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        from langchain_agentkit.extensions.filesystem.extension import (
            _wrap_with_permission_check,
        )

        class _Schema(BaseModel):
            foo: str

        async def _impl(foo: str) -> str:
            return foo

        inner = StructuredTool.from_function(
            coroutine=_impl,
            name="StubTool",
            description="stub",
            args_schema=_Schema,
            handle_tool_error=True,
        )

        wrapped = _wrap_with_permission_check(
            tool=inner,
            operation="write",
            target_arg="",  # unconfigured tool — _TOOL_TARGET_ARG.get fallthrough
            permissions=PERMISSIVE_RULESET,
            hitl_check=lambda: False,
        )

        result = await wrapped.ainvoke({"foo": "x"})

        assert "without a target argument" in result


# ---------------------------------------------------------------------------
# Prompt — Bash visibility
# ---------------------------------------------------------------------------


class TestPromptWithPermissions:
    async def test_prompt_does_not_reference_tool_names(self):
        """Tool names live on tool descriptions, not in the prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir, permissions=PERMISSIVE_RULESET)
            await ext.setup(extensions=[])
            result = ext.prompt({}, _TEST_RUNTIME) or ""
            for name in ("Read", "Write", "Edit", "Glob", "Grep", "Bash"):
                assert name not in result
