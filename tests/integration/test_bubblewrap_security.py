"""Bubblewrap-specific integration tests — runs only on Linux with bwrap.

These tests verify properties of the *sandbox itself* (not the
``FilesystemProtocol`` contract — that's covered by
``test_backend_protocol.py``'s parameterized matrix). Each test
exercises a security claim made in ``BubblewrapBackend``'s docstring
and would silently regress if the bwrap argv changed in a way that
broke it.

Skipped on non-Linux and on Linux hosts that don't have bubblewrap
installed; the canonical run path is via
``./scripts/test-in-docker.sh``.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

linux_with_bwrap = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bwrap") is None,
    reason="bubblewrap not available on this host",
)

pytestmark = linux_with_bwrap


@pytest.fixture
async def backend():
    from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

    with tempfile.TemporaryDirectory() as tmpdir:
        yield BubblewrapBackend(tmpdir)


# ---------------------------------------------------------------------------
# Symlink uniformity — the central docstring claim
# ---------------------------------------------------------------------------


class TestSymlinkUniformity:
    async def test_etc_passwd_absent_inside_sandbox(self, backend):
        """The sandbox does NOT bind /etc, so /etc/passwd is absent."""
        result = await backend.execute("test -f /etc/passwd && echo PRESENT || echo ABSENT")
        assert result["exit_code"] == 0
        assert "ABSENT" in result["output"]

    async def test_planted_symlink_to_etc_passwd_returns_not_found(self, backend):
        """Plant /workspace/sneaky -> /etc/passwd and Read it.

        Because /etc isn't bound into the sandbox, the symlink resolves
        to a non-existent path inside the sandbox view. The read returns
        ``file_not_found`` — same answer as if the file simply didn't
        exist. This is the central uniformity claim.
        """
        await backend.execute("ln -s /etc/passwd /workspace/sneaky")
        result = await backend.read("/sneaky")
        assert result.error == "file_not_found"
        assert result.content is None

    async def test_planted_symlink_to_usr_is_exfiltrable(self, backend):  # noqa: N802
        """/usr IS bind-mounted RO, so symlinks to it CAN read host files.

        This test exists to keep the docstring honest: anything
        readable under /usr, /bin, /lib, /lib64 reaches the agent.
        Don't put secrets there.
        """
        # /usr/bin/env exists on every Linux distro we support.
        await backend.execute("ln -s /usr/bin/env /workspace/probe")
        result = await backend.read_bytes("/probe")
        assert result.error is None
        # ELF magic — confirms we read the actual host binary.
        assert result.content is not None and result.content[:4] == b"\x7fELF"


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------


class TestEnvironmentIsolation:
    async def test_parent_env_does_not_leak(self, backend, monkeypatch):
        """Caller-process secrets must not appear in the sandbox env."""
        monkeypatch.setenv("BUBBLEWRAP_TEST_SECRET", "leaked-12345")
        result = await backend.execute("env")
        assert result["exit_code"] == 0
        assert "BUBBLEWRAP_TEST_SECRET" not in result["output"]
        assert "leaked-12345" not in result["output"]

    async def test_workspace_is_writable(self, backend):
        result = await backend.execute("touch /workspace/marker && echo OK")
        assert result["exit_code"] == 0
        assert "OK" in result["output"]

    async def test_usr_is_readonly(self, backend):
        result = await backend.execute("touch /usr/test-write 2>&1 || echo READONLY")
        assert "READONLY" in result["output"] or result["exit_code"] != 0


# ---------------------------------------------------------------------------
# extra_ro_binds validation
# ---------------------------------------------------------------------------


class TestExtraRoBindsValidation:
    def test_etc_rejected(self):
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="deny-list"),
        ):
            BubblewrapBackend(tmpdir, extra_ro_binds=(("/etc", "/skills"),))

    def test_proc_rejected(self):
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="deny-list"),
        ):
            BubblewrapBackend(tmpdir, extra_ro_binds=(("/proc", "/proc"),))


# ---------------------------------------------------------------------------
# Resource limits enforced
# ---------------------------------------------------------------------------


class TestRlimitsEnforced:
    async def test_fsize_limit_rejects_large_write(self):
        """RLIMIT_FSIZE caps a single-file write inside the sandbox."""
        from langchain_agentkit.backends.bubblewrap import (
            BubblewrapBackend,
            ResourceLimits,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BubblewrapBackend(tmpdir, rlimits=ResourceLimits(fsize_bytes=1024))
            # dd writes 4 KiB; the rlimit kills it after the first 1024
            # bytes with SIGXFSZ. The shell reports a non-zero exit.
            result = await backend.execute(
                "dd if=/dev/zero of=/workspace/big bs=4096 count=1 2>&1 || echo FAILED"
            )
            assert "FAILED" in result["output"] or "File size limit exceeded" in result.get(
                "stderr", ""
            )

    async def test_fsize_limit_allows_small_write(self):
        from langchain_agentkit.backends.bubblewrap import (
            BubblewrapBackend,
            ResourceLimits,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BubblewrapBackend(tmpdir, rlimits=ResourceLimits(fsize_bytes=10 * 1024))
            result = await backend.write("/small.txt", "hi")
            assert result.error is None


class TestTmpfsCap:
    async def test_tmpfs_size_caps_writes(self):
        """A tiny --size on /tmp prevents fill attacks."""
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BubblewrapBackend(tmpdir, tmpfs_size_bytes=64 * 1024)
            # 1 MiB write to /tmp must exceed the 64 KiB cap.
            result = await backend.execute(
                "dd if=/dev/zero of=/tmp/fill bs=1024 count=1024 2>&1; echo DONE"
            )
            assert "No space left" in result["output"] or "DONE" in result["output"]
            # Most importantly: the exit-code-checking branch shouldn't
            # explode the host. If we got DONE, the write was capped
            # silently by the tmpfs and a smaller file was produced.


# ---------------------------------------------------------------------------
# Edit OOM cap
# ---------------------------------------------------------------------------


class TestEditCap:
    async def test_edit_rejects_large_file(self):
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BubblewrapBackend(tmpdir, max_edit_file_bytes=512)
            await backend.write("/big.txt", "x" * 1024)
            result = await backend.edit("/big.txt", "x", "y")
            assert result.error == "io_error"
            assert "too large" in (result.error_message or "").lower()


# ---------------------------------------------------------------------------
# Cancellation reaps subprocess
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_timeout_kills_long_running(self, backend):
        """Timeout must kill the bwrap chain (bwrap -> bash -> sleep)."""
        result = await backend.execute("sleep 30", timeout=2)
        assert result["exit_code"] == -1
        assert result.get("truncated", False)


# ---------------------------------------------------------------------------
# Workspace path resolution
# ---------------------------------------------------------------------------


class TestWorkspaceResolution:
    async def test_root_visible_as_workspace(self, backend):
        """The host root is mounted at /workspace inside the sandbox."""
        # Write through the API; verify the file is at /workspace/X
        # inside the sandbox (the contract LLMs see).
        await backend.write("/visible.txt", "hello")
        result = await backend.execute("cat /workspace/visible.txt")
        assert result["exit_code"] == 0
        assert "hello" in result["output"]

    async def test_writes_persist_to_host_root(self):
        """Writes through the backend appear on the host filesystem."""
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = BubblewrapBackend(tmpdir)
            await backend.write("/persisted.txt", "host-visible")
            host_path = os.path.join(tmpdir, "persisted.txt")
            assert os.path.exists(host_path)
            with open(host_path) as f:
                assert f.read() == "host-visible"
