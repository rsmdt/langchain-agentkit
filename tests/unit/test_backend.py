"""Tests for BackendProtocol, SandboxProtocol, and OSBackend."""

from __future__ import annotations

import tempfile

import pytest

from langchain_agentkit.backend import (
    BackendProtocol,
    OSBackend,
)

# --- Protocol tests ---


class TestProtocolChecks:
    """Test runtime_checkable protocol conformance."""

    def test_os_backend_is_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert isinstance(backend, BackendProtocol)

    def test_os_backend_supports_execute(self):
        """OSBackend now implements the full 6-method protocol including execute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            # All backends now support execute (unified protocol)
            assert isinstance(backend, BackendProtocol)
            assert hasattr(backend, "execute")


# --- OSBackend tests ---


class TestOSBackend:
    """Test OSBackend with real filesystem."""

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "hello local")
            content = backend.read("/test.txt")
            assert "hello local" in content

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(PermissionError, match="Path traversal"):
                backend.read("/../../etc/passwd")

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            assert not backend.exists("/test.txt")
            backend.write("/test.txt", "data")
            assert backend.exists("/test.txt")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "data")
            backend.delete("/test.txt")
            assert not backend.exists("/test.txt")

    def test_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "hello world")
            result = backend.edit("/test.txt", "hello", "goodbye")
            assert result["replacements"] == 1
            content = backend.read("/test.txt")
            assert "goodbye world" in content

    def test_ls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/a.txt", "a")
            backend.write("/b.txt", "b")
            entries = backend.ls("/")
            paths = [e["path"] for e in entries]
            assert any("a.txt" in p for p in paths)
            assert any("b.txt" in p for p in paths)

    def test_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/src/a.py", "a")
            backend.write("/src/b.py", "b")
            backend.write("/src/c.txt", "c")
            matches = backend.glob("**/*.py")
            assert len(matches) >= 2

    def test_grep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "line one\nline two\nline three")
            matches = backend.grep("two", path="/")
            assert len(matches) >= 1

    def test_read_with_offset_and_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "line1\nline2\nline3\nline4\nline5\n")
            content = backend.read("/test.txt", offset=1, limit=2)
            assert "2\tline2" in content
            assert "3\tline3" in content
            assert "line1" not in content
            assert "line4" not in content

    def test_write_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/a/b/c/deep.txt", "deep")
            assert backend.exists("/a/b/c/deep.txt")

    def test_edit_replace_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/test.txt", "foo bar foo baz foo")
            result = backend.edit("/test.txt", "foo", "qux", replace_all=True)
            assert result["replacements"] == 3
            content = backend.read("/test.txt")
            assert "qux bar qux baz qux" in content

    def test_delete_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/dir/a.txt", "a")
            backend.write("/dir/b.txt", "b")
            backend.delete("/dir")
            assert not backend.exists("/dir")

    def test_ls_returns_size_and_is_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/file.txt", "hello")
            backend.write("/sub/nested.txt", "data")
            entries = backend.ls("/")
            file_entry = next(e for e in entries if "file.txt" in e["path"])
            dir_entry = next(e for e in entries if "sub" in e["path"])
            assert file_entry["size"] == 5
            assert not file_entry["is_dir"]
            assert dir_entry["is_dir"]

    def test_write_returns_bytes_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            result = backend.write("/test.txt", "hello")
            assert result["path"] == "/test.txt"
            assert result["bytes_written"] == 5

    def test_read_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/empty.txt", "")
            content = backend.read("/empty.txt")
            assert content == ""

    def test_read_nonexistent_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(FileNotFoundError):
                backend.read("/nope.txt")

    def test_edit_nonexistent_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(FileNotFoundError):
                backend.edit("/nope.txt", "a", "b")

    def test_write_binary_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            data = b"\x89PNG\r\n\x1a\n\x00\x00"
            result = backend.write("/image.bin", data)
            assert result["bytes_written"] == len(data)
            assert backend.exists("/image.bin")

    def test_read_offset_beyond_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/short.txt", "one\ntwo\n")
            content = backend.read("/short.txt", offset=100)
            assert content == ""

    def test_grep_ignore_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/f.txt", "Hello World\ngoodbye")
            matches = backend.grep("hello", ignore_case=True)
            assert len(matches) == 1
            assert "Hello World" in matches[0]["text"]

    def test_grep_case_sensitive_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/f.txt", "Hello World")
            matches = backend.grep("hello")
            assert len(matches) == 0

    def test_grep_skips_binary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/bin.dat", b"\x00\x01\x02\x03")
            backend.write("/text.txt", "findme")
            matches = backend.grep("findme")
            assert len(matches) == 1

    def test_ls_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            entries = backend.ls("/")
            assert entries == []

    def test_ls_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            entries = backend.ls("/nonexistent")
            assert entries == []

    def test_glob_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            matches = backend.glob("*.xyz")
            assert matches == []

    def test_delete_nonexistent_is_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            # Should not raise
            backend.delete("/nope.txt")

    def test_symlink_inside_root_allowed(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/real.txt", "content")
            os.symlink(
                os.path.join(tmpdir, "real.txt"),
                os.path.join(tmpdir, "link.txt"),
            )
            content = backend.read("/link.txt")
            assert "content" in content

    def test_symlink_outside_root_blocked(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            # Create symlink pointing outside root
            os.symlink("/etc/hosts", os.path.join(tmpdir, "escape.txt"))
            with pytest.raises(PermissionError, match="Path traversal"):
                backend.read("/escape.txt")

    def test_edit_returns_zero_when_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/f.txt", "hello")
            result = backend.edit("/f.txt", "missing", "x")
            assert result["replacements"] == 0

    def test_read_bytes_returns_raw_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            backend.write("/image.png", data)
            result = backend.read_bytes("/image.png")
            assert result == data

    def test_read_bytes_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/hello.txt", "hello world")
            result = backend.read_bytes("/hello.txt")
            assert result == b"hello world"

    def test_read_bytes_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(FileNotFoundError):
                backend.read_bytes("/nope.bin")

    def test_read_bytes_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            with pytest.raises(PermissionError, match="Path traversal"):
                backend.read_bytes("/../../etc/passwd")

    def test_unicode_content_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            backend.write("/unicode.txt", "日本語\n中文\nعربي\n")
            content = backend.read("/unicode.txt")
            assert "日本語" in content
            assert "中文" in content
            assert "عربي" in content
