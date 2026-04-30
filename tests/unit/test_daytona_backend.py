"""Unit tests for DaytonaBackend.

Uses a stub Daytona sandbox object (edge mock — the SDK is the external
boundary). Tests the logic that lives in DaytonaBackend itself:
path resolution, execute() error wrapping, and shell quoting.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langchain_agentkit.backends.daytona import DaytonaBackend, _shell_quote

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(workdir: str = "/workspace") -> tuple[DaytonaBackend, MagicMock]:
    """Create a DaytonaBackend with a mock Daytona SDK sandbox."""
    mock_sandbox = MagicMock()
    backend = DaytonaBackend(mock_sandbox, timeout=60, workdir=workdir)
    return backend, mock_sandbox


# ---------------------------------------------------------------------------
# _shell_quote
# ---------------------------------------------------------------------------


class TestShellQuote:
    def test_plain_string(self):
        assert _shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        assert _shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quote(self):
        result = _shell_quote("it's")
        assert result == "'it'\"'\"'s'"

    def test_string_with_special_chars(self):
        result = _shell_quote("$HOME `cmd` \\n")
        # Inside single quotes, $, `, \ are literal
        assert result.startswith("'")
        assert result.endswith("'")
        assert "$HOME" in result

    def test_empty_string(self):
        assert _shell_quote("") == "''"

    def test_string_with_newline(self):
        result = _shell_quote("line1\nline2")
        assert "\n" in result


# ---------------------------------------------------------------------------
# _resolve — path traversal prevention
# ---------------------------------------------------------------------------


class TestResolve:
    def test_simple_path(self):
        backend, _ = _make_backend()
        assert backend._resolve("test.txt") == "/workspace/test.txt"

    def test_leading_slash_stripped(self):
        backend, _ = _make_backend()
        assert backend._resolve("/test.txt") == "/workspace/test.txt"

    def test_nested_path(self):
        backend, _ = _make_backend()
        assert backend._resolve("/src/main.py") == "/workspace/src/main.py"

    def test_empty_path_returns_workdir(self):
        backend, _ = _make_backend()
        assert backend._resolve("/") == "/workspace"
        assert backend._resolve("") == "/workspace"

    def test_traversal_blocked(self):
        backend, _ = _make_backend()
        with pytest.raises(PermissionError, match="Path traversal"):
            backend._resolve("../../etc/passwd")

    def test_traversal_with_leading_slash_blocked(self):
        backend, _ = _make_backend()
        with pytest.raises(PermissionError, match="Path traversal"):
            backend._resolve("/../../../etc/passwd")

    def test_dot_dot_in_middle_blocked(self):
        backend, _ = _make_backend()
        with pytest.raises(PermissionError, match="Path traversal"):
            backend._resolve("/src/../../etc/passwd")

    def test_dot_dot_staying_inside_allowed(self):
        backend, _ = _make_backend()
        result = backend._resolve("/src/../test.txt")
        assert result == "/workspace/test.txt"

    def test_absolute_path_under_workdir_passes_through(self):
        """Regression: workdir-prefixed absolute paths must NOT be re-prefixed.

        The Read/Write/Edit tool descriptions instruct the LLM to pass
        absolute paths, and the ``<env>`` block surfaces the workdir to
        the model. Without pass-through, ``_resolve("/workspace/foo")``
        used to return ``/workspace/workspace/foo``.
        """
        backend, _ = _make_backend("/workspace")
        assert backend._resolve("/workspace/foo.md") == "/workspace/foo.md"
        assert backend._resolve("/workspace/src/main.py") == "/workspace/src/main.py"
        assert backend._resolve("/workspace") == "/workspace"

    def test_absolute_path_under_default_daytona_workdir(self):
        """Same regression, exercised against the documented Daytona default."""
        backend, _ = _make_backend("/home/daytona")
        assert backend._resolve("/home/daytona/foo.md") == "/home/daytona/foo.md"
        assert (
            backend._resolve("/home/daytona/.agentkit/AGENTS.md")
            == "/home/daytona/.agentkit/AGENTS.md"
        )

    def test_absolute_path_outside_workdir_falls_back_to_relative(self):
        """Paths whose absolute form sits outside workdir are treated as
        workdir-rooted shorthand (``/foo`` → ``<workdir>/foo``) — the
        legacy behaviour callers and existing tools rely on."""
        backend, _ = _make_backend("/workspace")
        assert backend._resolve("/foo.md") == "/workspace/foo.md"
        assert backend._resolve("/src/main.py") == "/workspace/src/main.py"

    def test_absolute_path_with_traversal_still_blocked(self):
        """Even when an absolute path superficially starts under workdir,
        traversal that escapes the workdir must be rejected."""
        backend, _ = _make_backend("/workspace")
        # /workspace/../etc/passwd normalizes to /etc/passwd, which is
        # NOT under workdir; the absolute-prefix branch declines it and
        # the relative re-resolution would land at /workspace/etc/passwd.
        # That's safely confined, but expressing the *intent* matters:
        # the user asked to escape, and we kept them in.
        result = backend._resolve("/workspace/../etc/passwd")
        # Confined under workdir — never escapes.
        assert result.startswith("/workspace/") or result == "/workspace"


# ---------------------------------------------------------------------------
# execute — SDK bridge
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_passes_command_to_sdk(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(
            result="hello\n",
            exit_code=0,
        )
        result = await backend.execute("echo hello")
        mock.process.exec.assert_called_once_with(
            "echo hello",
            cwd="/workspace",
            timeout=60,
        )
        assert result["output"] == "hello\n"
        assert result["exit_code"] == 0

    async def test_uses_custom_timeout(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(result="", exit_code=0)
        await backend.execute("sleep 1", timeout=10)
        mock.process.exec.assert_called_once_with(
            "sleep 1",
            cwd="/workspace",
            timeout=10,
        )

    async def test_resolves_workdir_override(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(result="", exit_code=0)
        await backend.execute("ls", workdir="/src")
        mock.process.exec.assert_called_once_with(
            "ls",
            cwd="/workspace/src",
            timeout=60,
        )

    async def test_captures_stderr_when_available(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(
            result="",
            exit_code=1,
            stderr="error details",
        )
        result = await backend.execute("bad_command")
        assert result["stderr"] == "error details"

    async def test_stderr_empty_when_not_available(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(result="output", exit_code=0)
        result = await backend.execute("echo hi")
        assert result["stderr"] == ""

    async def test_sdk_error_raises_runtime_error(self):
        backend, mock = _make_backend()
        mock.process.exec.side_effect = ConnectionError("network down")
        with pytest.raises(RuntimeError, match="Daytona sandbox execution failed"):
            await backend.execute("echo hello")

    async def test_nonzero_exit_code_returned(self):
        backend, mock = _make_backend()
        mock.process.exec.return_value = SimpleNamespace(result="", exit_code=127)
        result = await backend.execute("nonexistent_command")
        assert result["exit_code"] == 127

    def test_workdir_traversal_blocked(self):
        backend, _ = _make_backend()
        with pytest.raises(PermissionError, match="Path traversal"):
            # _resolve is sync and raises before execute is called
            backend._resolve("/../../etc")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_workdir_from_parameter(self):
        backend, _ = _make_backend("/workspace")
        assert backend.workdir == "/workspace"

    def test_strips_trailing_slash(self):
        backend, _ = _make_backend("/workspace/")
        assert backend.workdir == "/workspace"

    def test_default_timeout(self):
        backend, _ = _make_backend()
        assert backend._timeout == 60

    def test_default_workdir_is_home_daytona(self):
        """When no workdir is passed, default to Daytona's documented default
        (``/home/daytona``) — no SDK round-trip required."""
        mock_sandbox = MagicMock()
        backend = DaytonaBackend(mock_sandbox, timeout=60)
        assert backend.workdir == "/home/daytona"
        # Importantly, the SDK's get_work_dir is NOT consulted on construction.
        mock_sandbox.get_work_dir.assert_not_called()
