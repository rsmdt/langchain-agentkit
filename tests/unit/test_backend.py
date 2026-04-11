"""OSBackend-specific tests.

Tests for behavior unique to OSBackend that is NOT part of BackendProtocol:
symlinks, absolute path resolution, ls(), exists(), delete().

Protocol-level tests (read, write, edit, glob, grep, execute) live in
test_backend_protocol.py and run against all backends via matrix.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from langchain_agentkit.backends import BackendProtocol, OSBackend


class TestProtocolConformance:
    def test_is_backend_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert isinstance(OSBackend(tmpdir), BackendProtocol)

    def test_has_execute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert hasattr(backend, "execute") and callable(backend.execute)


class TestExecuteTimeout:
    async def test_timeout_returns_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            result = await backend.execute("sleep 10", timeout=1)
            assert result["exit_code"] == -1
            assert result["truncated"] is True


class TestAbsolutePathResolution:
    """OSBackend resolves absolute paths that are already inside root."""

    async def test_absolute_path_inside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/workspace/config.json", '{"key": "value"}')
            abs_path = os.path.join(os.path.realpath(tmpdir), "workspace/config.json")
            content = await backend.read(abs_path)
            assert "value" in content

    async def test_absolute_path_outside_root_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(PermissionError, match="Path traversal"):
                await backend.read("/../../../etc/passwd")

    async def test_absolute_path_truly_outside_root_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as other:
            other_file = os.path.join(other, "secret.txt")
            with open(other_file, "w") as f:
                f.write("secret")
            backend = OSBackend(tmpdir)
            with pytest.raises((PermissionError, FileNotFoundError)):
                await backend.read(other_file)


class TestSymlinks:
    async def test_symlink_inside_root_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/real.txt", "content")
            os.symlink(
                os.path.join(tmpdir, "real.txt"),
                os.path.join(tmpdir, "link.txt"),
            )
            content = await backend.read("/link.txt")
            assert "content" in content

    async def test_symlink_outside_root_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            os.symlink("/etc/hosts", os.path.join(tmpdir, "escape.txt"))
            with pytest.raises(PermissionError, match="Path traversal"):
                await backend.read("/escape.txt")


class TestConvenienceMethods:
    """ls(), exists(), delete() — not part of BackendProtocol."""

    async def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert not backend.exists("/test.txt")
            await backend.write("/test.txt", "data")
            assert backend.exists("/test.txt")

    async def test_delete_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/test.txt", "data")
            backend.delete("/test.txt")
            assert not backend.exists("/test.txt")

    async def test_delete_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/dir/a.txt", "a")
            await backend.write("/dir/b.txt", "b")
            backend.delete("/dir")
            assert not backend.exists("/dir")

    def test_delete_nonexistent_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.delete("/nope.txt")  # should not raise

    async def test_ls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/a.txt", "a")
            await backend.write("/b.txt", "b")
            entries = backend.ls("/")
            paths = [e["path"] for e in entries]
            assert any("a.txt" in p for p in paths)
            assert any("b.txt" in p for p in paths)

    async def test_ls_returns_size_and_is_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            await backend.write("/file.txt", "hello")
            await backend.write("/sub/nested.txt", "data")
            entries = backend.ls("/")
            file_entry = next(e for e in entries if "file.txt" in e["path"])
            dir_entry = next(e for e in entries if "sub" in e["path"])
            assert file_entry["size"] == 5
            assert not file_entry["is_dir"]
            assert dir_entry["is_dir"]

    def test_ls_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert backend.ls("/") == []

    def test_ls_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert backend.ls("/nonexistent") == []
