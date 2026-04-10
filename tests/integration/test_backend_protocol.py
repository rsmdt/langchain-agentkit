"""BackendProtocol conformance tests — run against every backend.

These tests verify the ``BackendProtocol`` contract: read, read_bytes,
write, edit, glob, grep, execute. Any backend that implements the
protocol must pass all of these.

The ``backend`` fixture is parameterized so the same tests run against
every available backend:

- ``os`` — always available (OSBackend with temp directory)
- ``daytona`` — only when DAYTONA_API_URL is set and SDK is installed

To add a new backend, add a case to the ``backend`` fixture below.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from langchain_agentkit.backends import BackendProtocol, OSBackend

# ---------------------------------------------------------------------------
# .env loading (for DAYTONA_API_URL / DAYTONA_API_KEY)
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Daytona helpers
# ---------------------------------------------------------------------------


def _import_daytona():
    """Import Daytona SDK — supports both package names."""
    try:
        from daytona_sdk import Daytona, DaytonaConfig  # type: ignore[import-not-found]

        return Daytona, DaytonaConfig
    except ImportError:
        pass
    try:
        from daytona import Daytona, DaytonaConfig  # type: ignore[import-not-found]

        return Daytona, DaytonaConfig
    except ImportError:
        return None, None


def _daytona_available() -> bool:
    if not os.environ.get("DAYTONA_API_URL"):
        return False
    cls, _ = _import_daytona()
    return cls is not None


# ---------------------------------------------------------------------------
# Parameterized backend fixture
# ---------------------------------------------------------------------------

_BACKEND_IDS = ["os"]
if _daytona_available():
    _BACKEND_IDS.append("daytona")


@pytest.fixture(params=_BACKEND_IDS)
def backend(request):
    """Yield a fresh backend instance for each parameterized backend type."""
    if request.param == "os":
        with tempfile.TemporaryDirectory() as tmpdir:
            yield OSBackend(tmpdir)

    elif request.param == "daytona":
        daytona_cls, config_cls = _import_daytona()
        from langchain_agentkit.backends.daytona import DaytonaBackend

        api_key = os.environ.get("DAYTONA_API_KEY", "")
        config = config_cls(
            api_key=api_key or None,
            api_url=os.environ["DAYTONA_API_URL"],
        )
        client = daytona_cls(config=config)
        sandbox = client.create()
        yield DaytonaBackend(sandbox)
        sandbox.delete()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_is_backend_protocol(self, backend):
        assert isinstance(backend, BackendProtocol)

    def test_has_execute(self, backend):
        assert hasattr(backend, "execute") and callable(backend.execute)


# ---------------------------------------------------------------------------
# read / read_bytes
# ---------------------------------------------------------------------------


class TestRead:
    def test_write_and_read_roundtrip(self, backend):
        backend.write("/test.txt", "hello world")
        content = backend.read("/test.txt")
        assert "hello world" in content

    def test_returns_raw_text(self, backend):
        backend.write("/test.txt", "alpha\nbeta\ngamma\n")
        content = backend.read("/test.txt")
        assert content == "alpha\nbeta\ngamma\n"

    def test_offset_and_limit(self, backend):
        backend.write("/test.txt", "line1\nline2\nline3\nline4\nline5\n")
        content = backend.read("/test.txt", offset=1, limit=2)
        assert "line2" in content
        assert "line3" in content
        assert "line1" not in content
        assert "line4" not in content

    def test_offset_beyond_file(self, backend):
        backend.write("/short.txt", "one\ntwo\n")
        content = backend.read("/short.txt", offset=100)
        assert content == ""

    def test_empty_file(self, backend):
        backend.write("/empty.txt", "")
        content = backend.read("/empty.txt")
        assert content == ""

    def test_missing_file_raises(self, backend):
        with pytest.raises(FileNotFoundError):
            backend.read("/nonexistent.txt")

    def test_unicode_roundtrip(self, backend):
        backend.write("/uni.txt", "日本語\n中文\nعربي\n")
        content = backend.read("/uni.txt")
        assert "日本語" in content
        assert "中文" in content
        assert "عربي" in content


class TestReadBytes:
    def test_returns_raw_content(self, backend):
        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        backend.write("/image.png", data)
        assert backend.read_bytes("/image.png") == data

    def test_text_file(self, backend):
        backend.write("/hello.txt", "hello world")
        assert backend.read_bytes("/hello.txt") == b"hello world"

    def test_missing_file_raises(self, backend):
        with pytest.raises(FileNotFoundError):
            backend.read_bytes("/nonexistent.bin")

    def test_path_traversal_blocked(self, backend):
        with pytest.raises(PermissionError, match="Path traversal"):
            backend.read_bytes("/../../etc/passwd")


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestWrite:
    def test_returns_bytes_written(self, backend):
        result = backend.write("/test.txt", "hello")
        assert result["path"] == "/test.txt"
        assert result["bytes_written"] == 5

    def test_creates_parent_dirs(self, backend):
        backend.write("/a/b/c/deep.txt", "deep")
        content = backend.read("/a/b/c/deep.txt")
        assert "deep" in content

    def test_binary_content(self, backend):
        data = b"\x89PNG\r\n\x1a\n\x00\x00"
        result = backend.write("/image.bin", data)
        assert result["bytes_written"] == len(data)


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    def test_single_replacement(self, backend):
        backend.write("/test.txt", "hello world")
        result = backend.edit("/test.txt", "hello", "goodbye")
        assert result["replacements"] == 1
        content = backend.read("/test.txt")
        assert "goodbye world" in content

    def test_replace_all(self, backend):
        backend.write("/test.txt", "foo bar foo baz foo")
        result = backend.edit("/test.txt", "foo", "qux", replace_all=True)
        assert result["replacements"] == 3
        content = backend.read("/test.txt")
        assert "qux bar qux baz qux" in content

    def test_ambiguous_raises(self, backend):
        backend.write("/test.txt", "foo bar foo")
        with pytest.raises(ValueError, match="Ambiguous"):
            backend.edit("/test.txt", "foo", "baz")

    def test_not_found_returns_zero(self, backend):
        backend.write("/f.txt", "hello")
        result = backend.edit("/f.txt", "missing", "x")
        assert result["replacements"] == 0

    def test_missing_file_raises(self, backend):
        with pytest.raises((FileNotFoundError, ValueError)):
            backend.edit("/nonexistent.txt", "a", "b")


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestGlob:
    def test_finds_files(self, backend):
        backend.write("/src/a.py", "a")
        backend.write("/src/b.py", "b")
        backend.write("/src/c.txt", "c")
        matches = backend.glob("**/*.py")
        py_files = [m for m in matches if m.endswith(".py")]
        assert len(py_files) >= 2

    def test_no_matches(self, backend):
        matches = backend.glob("*.xyz")
        assert matches == []


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_finds_pattern(self, backend):
        backend.write("/test.txt", "line one\nline two\nline three")
        matches = backend.grep("two", path="/")
        assert len(matches) >= 1

    def test_ignore_case(self, backend):
        backend.write("/f.txt", "Hello World\ngoodbye")
        matches = backend.grep("hello", ignore_case=True)
        assert len(matches) == 1
        assert "Hello World" in matches[0]["text"]

    def test_case_sensitive_by_default(self, backend):
        backend.write("/f.txt", "Hello World")
        matches = backend.grep("hello")
        assert len(matches) == 0

    def test_no_matches(self, backend):
        backend.write("/f.txt", "nothing here")
        matches = backend.grep("missing")
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_echo(self, backend):
        result = backend.execute("echo hello")
        assert result["exit_code"] == 0
        assert "hello" in result["output"]

    def test_nonzero_exit(self, backend):
        result = backend.execute("exit 1")
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# security
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_path_traversal_blocked(self, backend):
        with pytest.raises(PermissionError):
            backend.read("/../../etc/passwd")
