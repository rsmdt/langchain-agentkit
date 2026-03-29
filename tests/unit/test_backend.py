"""Tests for BackendProtocol, SandboxProtocol, and implementations."""

from __future__ import annotations

import os
import tempfile

import pytest

from langchain_agentkit.backend import (
    BackendProtocol,
    CompositeBackend,
    LocalBackend,
    MemoryBackend,
    SandboxProtocol,
)


# --- Protocol tests ---


class TestProtocolChecks:
    """Test runtime_checkable protocol conformance."""

    def test_memory_backend_is_backend(self):
        backend = MemoryBackend()
        assert isinstance(backend, BackendProtocol)

    def test_memory_backend_is_not_sandbox(self):
        backend = MemoryBackend()
        assert not isinstance(backend, SandboxProtocol)

    def test_local_backend_is_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            assert isinstance(backend, BackendProtocol)

    def test_composite_backend_is_backend(self):
        backend = CompositeBackend(default=MemoryBackend())
        assert isinstance(backend, BackendProtocol)


# --- MemoryBackend tests ---


class TestMemoryBackend:
    """Test MemoryBackend wrapping VirtualFilesystem."""

    def test_write_and_read(self):
        backend = MemoryBackend()
        result = backend.write("/test.txt", "hello world")
        assert result["bytes_written"] == 11

        content = backend.read("/test.txt")
        assert "hello world" in content

    def test_exists(self):
        backend = MemoryBackend()
        assert not backend.exists("/test.txt")
        backend.write("/test.txt", "data")
        assert backend.exists("/test.txt")

    def test_delete(self):
        backend = MemoryBackend()
        backend.write("/test.txt", "data")
        assert backend.exists("/test.txt")
        backend.delete("/test.txt")
        assert not backend.exists("/test.txt")

    def test_edit(self):
        backend = MemoryBackend()
        backend.write("/test.txt", "hello world")
        result = backend.edit("/test.txt", "hello", "goodbye")
        assert result["replacements"] == 1
        content = backend.read("/test.txt")
        assert "goodbye world" in content

    def test_ls(self):
        backend = MemoryBackend()
        backend.write("/dir/a.txt", "a")
        backend.write("/dir/b.txt", "b")
        entries = backend.ls("/dir")
        paths = [e["path"] for e in entries]
        assert "/dir/a.txt" in paths
        assert "/dir/b.txt" in paths

    def test_glob(self):
        backend = MemoryBackend()
        backend.write("/src/a.py", "a")
        backend.write("/src/b.py", "b")
        backend.write("/src/c.txt", "c")
        matches = backend.glob("*.py", path="/src")
        assert len(matches) == 2

    def test_grep(self):
        backend = MemoryBackend()
        backend.write("/test.txt", "line one\nline two\nline three")
        matches = backend.grep("two")
        assert len(matches) >= 1
        assert any("two" in m["text"] for m in matches)


# --- LocalBackend tests ---


class TestLocalBackend:
    """Test LocalBackend with real filesystem."""

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/test.txt", "hello local")
            content = backend.read("/test.txt")
            assert "hello local" in content

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            with pytest.raises(PermissionError, match="Path traversal"):
                backend.read("/../../etc/passwd")

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            assert not backend.exists("/test.txt")
            backend.write("/test.txt", "data")
            assert backend.exists("/test.txt")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/test.txt", "data")
            backend.delete("/test.txt")
            assert not backend.exists("/test.txt")

    def test_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/test.txt", "hello world")
            result = backend.edit("/test.txt", "hello", "goodbye")
            assert result["replacements"] == 1
            content = backend.read("/test.txt")
            assert "goodbye world" in content

    def test_ls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/a.txt", "a")
            backend.write("/b.txt", "b")
            entries = backend.ls("/")
            paths = [e["path"] for e in entries]
            assert any("a.txt" in p for p in paths)
            assert any("b.txt" in p for p in paths)

    def test_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/src/a.py", "a")
            backend.write("/src/b.py", "b")
            backend.write("/src/c.txt", "c")
            matches = backend.glob("**/*.py")
            assert len(matches) >= 2

    def test_grep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalBackend(tmpdir)
            backend.write("/test.txt", "line one\nline two\nline three")
            matches = backend.grep("two", path="/")
            assert len(matches) >= 1


# --- CompositeBackend tests ---


class TestCompositeBackend:
    """Test CompositeBackend with prefix routing."""

    def test_routes_to_correct_backend(self):
        mem_a = MemoryBackend()
        mem_b = MemoryBackend()

        composite = CompositeBackend(
            default=mem_a,
            routes={"/workspace/": mem_b},
        )

        composite.write("/workspace/test.txt", "in workspace")
        composite.write("/other.txt", "in default")

        # workspace file should be in mem_b
        assert mem_b.exists("/test.txt")
        assert not mem_a.exists("/test.txt")

        # other file should be in mem_a
        assert mem_a.exists("/other.txt")

    def test_longest_prefix_wins(self):
        mem_short = MemoryBackend()
        mem_long = MemoryBackend()

        composite = CompositeBackend(
            default=MemoryBackend(),
            routes={
                "/a/": mem_short,
                "/a/b/": mem_long,
            },
        )

        composite.write("/a/b/file.txt", "deep")

        assert mem_long.exists("/file.txt")
        assert not mem_short.exists("/b/file.txt")

    def test_read_from_routed_backend(self):
        mem = MemoryBackend()
        mem.write("/file.txt", "routed content")

        composite = CompositeBackend(
            default=MemoryBackend(),
            routes={"/data/": mem},
        )

        content = composite.read("/data/file.txt")
        assert "routed content" in content

    def test_default_backend_used_for_unmatched_paths(self):
        default = MemoryBackend()
        composite = CompositeBackend(default=default)

        composite.write("/anywhere.txt", "data")
        assert default.exists("/anywhere.txt")
