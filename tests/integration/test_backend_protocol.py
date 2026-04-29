"""BackendProtocol conformance tests — run against every backend.

These tests verify the contract of all three capability protocols:
``BackendProtocol`` (file ops), ``SandboxBackend`` (adds execute),
``FileTransferBackend`` (adds upload/download). Any backend that
implements a tier must pass the corresponding tests.

The ``backend`` fixture is parameterized so the same tests run against
every available backend:

- ``os`` — always available (OSBackend with temp directory)
- ``daytona`` — only when DAYTONA_API_URL is set and SDK is installed
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from langchain_agentkit.backends import (
    BackendProtocol,
    FileTransferBackend,
    OSBackend,
    SandboxBackend,
)

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
    """Daytona conformance runs only when both env vars and SDK are present."""
    if not os.environ.get("DAYTONA_API_URL"):
        return False
    if not os.environ.get("DAYTONA_API_KEY"):
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

        config = config_cls(
            api_key=os.environ["DAYTONA_API_KEY"],
            api_url=os.environ["DAYTONA_API_URL"],
        )
        client = daytona_cls(config=config)
        sandbox = client.create()
        yield DaytonaBackend(sandbox)
        sandbox.delete()


# ---------------------------------------------------------------------------
# Protocol tier conformance
# ---------------------------------------------------------------------------


class TestProtocolTiers:
    def test_implements_backend_protocol(self, backend):
        assert isinstance(backend, BackendProtocol)

    def test_implements_sandbox_backend(self, backend):
        assert isinstance(backend, SandboxBackend)

    def test_implements_file_transfer_backend(self, backend):
        assert isinstance(backend, FileTransferBackend)


# ---------------------------------------------------------------------------
# read / read_bytes
# ---------------------------------------------------------------------------


class TestRead:
    async def test_write_and_read_roundtrip(self, backend):
        await backend.write("/test.txt", "hello world")
        result = await backend.read("/test.txt")
        assert result.error is None
        assert "hello world" in result.content

    async def test_returns_raw_text(self, backend):
        await backend.write("/test.txt", "alpha\nbeta\ngamma\n")
        result = await backend.read("/test.txt")
        assert result.error is None
        assert result.content == "alpha\nbeta\ngamma\n"

    async def test_offset_and_limit(self, backend):
        await backend.write("/test.txt", "line1\nline2\nline3\nline4\nline5\n")
        result = await backend.read("/test.txt", offset=1, limit=2)
        assert result.error is None
        assert "line2" in result.content
        assert "line3" in result.content
        assert "line1" not in result.content
        assert "line4" not in result.content

    async def test_offset_beyond_file(self, backend):
        await backend.write("/short.txt", "one\ntwo\n")
        result = await backend.read("/short.txt", offset=100)
        assert result.error is None
        assert result.content == ""

    async def test_empty_file(self, backend):
        await backend.write("/empty.txt", "")
        result = await backend.read("/empty.txt")
        assert result.error is None
        assert result.content == ""

    async def test_missing_file_returns_error_code(self, backend):
        result = await backend.read("/nonexistent.txt")
        assert result.error == "file_not_found"
        assert result.content is None

    async def test_unicode_roundtrip(self, backend):
        await backend.write("/uni.txt", "日本語\n中文\nعربي\n")
        result = await backend.read("/uni.txt")
        assert result.error is None
        assert "日本語" in result.content
        assert "中文" in result.content
        assert "عربي" in result.content


class TestReadBytes:
    async def test_returns_raw_content(self, backend):
        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        await backend.write("/image.png", data)
        result = await backend.read_bytes("/image.png")
        assert result.error is None
        assert result.content == data

    async def test_text_file(self, backend):
        await backend.write("/hello.txt", "hello world")
        result = await backend.read_bytes("/hello.txt")
        assert result.error is None
        assert result.content == b"hello world"

    async def test_missing_file_returns_error_code(self, backend):
        result = await backend.read_bytes("/nonexistent.bin")
        assert result.error == "file_not_found"
        assert result.content is None

    async def test_path_traversal_returns_error_code(self, backend):
        result = await backend.read_bytes("/../../etc/passwd")
        assert result.error == "permission_denied"


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestWrite:
    async def test_returns_bytes_written(self, backend):
        result = await backend.write("/test.txt", "hello")
        assert result.error is None
        assert result.path == "/test.txt"
        assert result.bytes_written == 5

    async def test_creates_parent_dirs(self, backend):
        await backend.write("/a/b/c/deep.txt", "deep")
        result = await backend.read("/a/b/c/deep.txt")
        assert result.error is None
        assert "deep" in result.content

    async def test_binary_content(self, backend):
        data = b"\x89PNG\r\n\x1a\n\x00\x00"
        result = await backend.write("/image.bin", data)
        assert result.error is None
        assert result.bytes_written == len(data)


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class TestEdit:
    async def test_single_replacement(self, backend):
        await backend.write("/test.txt", "hello world")
        result = await backend.edit("/test.txt", "hello", "goodbye")
        assert result.error is None
        assert result.replacements == 1
        read = await backend.read("/test.txt")
        assert "goodbye world" in read.content

    async def test_replace_all(self, backend):
        await backend.write("/test.txt", "foo bar foo baz foo")
        result = await backend.edit("/test.txt", "foo", "qux", replace_all=True)
        assert result.error is None
        assert result.replacements == 3
        read = await backend.read("/test.txt")
        assert "qux bar qux baz qux" in read.content

    async def test_ambiguous_returns_error_code(self, backend):
        await backend.write("/test.txt", "foo bar foo")
        result = await backend.edit("/test.txt", "foo", "baz")
        assert result.error == "ambiguous_match"
        assert result.occurrences == 2
        assert result.replacements is None

    async def test_not_found_returns_error_code(self, backend):
        await backend.write("/f.txt", "hello")
        result = await backend.edit("/f.txt", "missing", "x")
        assert result.error == "old_string_not_found"
        assert result.replacements is None

    async def test_missing_file_returns_error_code(self, backend):
        result = await backend.edit("/nonexistent.txt", "a", "b")
        assert result.error == "file_not_found"


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestGlob:
    async def test_finds_files(self, backend):
        await backend.write("/src/a.py", "a")
        await backend.write("/src/b.py", "b")
        await backend.write("/src/c.txt", "c")
        matches = await backend.glob("**/*.py")
        py_files = [m for m in matches if m.endswith(".py")]
        assert len(py_files) >= 2

    async def test_no_matches(self, backend):
        matches = await backend.glob("*.xyz")
        assert matches == []


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    async def test_finds_pattern(self, backend):
        await backend.write("/test.txt", "line one\nline two\nline three")
        matches = await backend.grep("two", path="/")
        assert len(matches) >= 1

    async def test_ignore_case(self, backend):
        await backend.write("/f.txt", "Hello World\ngoodbye")
        matches = await backend.grep("hello", ignore_case=True)
        assert len(matches) == 1
        assert "Hello World" in matches[0]["text"]

    async def test_case_sensitive_by_default(self, backend):
        await backend.write("/f.txt", "Hello World")
        matches = await backend.grep("hello")
        assert len(matches) == 0

    async def test_no_matches(self, backend):
        await backend.write("/f.txt", "nothing here")
        matches = await backend.grep("missing")
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# execute (SandboxBackend)
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_echo(self, backend):
        result = await backend.execute("echo hello")
        assert result["exit_code"] == 0
        assert "hello" in result["output"]

    async def test_nonzero_exit(self, backend):
        result = await backend.execute("exit 1")
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# environment (SandboxBackend)
# ---------------------------------------------------------------------------


class TestEnvironment:
    async def test_returns_environment_snapshot(self, backend):
        from langchain_agentkit.backends import PROBED_TOOLS

        env = await backend.environment()
        # os encodes uname -srm shape: kernel-name + release + arch.
        # The leading word identifies the platform family unambiguously.
        assert env.os, "os must be a non-empty string"
        assert env.os.split()[0] in {"Linux", "Darwin", "Windows"}
        # cwd reflects backend root / workdir
        assert env.cwd
        # shell is a non-empty string (typically /bin/sh, /bin/bash, or /bin/zsh)
        assert env.shell
        # available_tools is a frozenset and a subset of PROBED_TOOLS
        # (no specific tool is required to be installed in test environments)
        assert isinstance(env.available_tools, frozenset)
        assert env.available_tools.issubset(set(PROBED_TOOLS))

    async def test_cached(self, backend):
        env1 = await backend.environment()
        env2 = await backend.environment()
        assert env1 is env2  # same object — backend caches


# ---------------------------------------------------------------------------
# upload_files / download_files (FileTransferBackend)
# ---------------------------------------------------------------------------


class TestFileTransfer:
    async def test_upload_files_roundtrip(self, backend):
        files = [
            ("/seed/a.txt", b"alpha"),
            ("/seed/b.bin", b"\x00\x01\x02\x03"),
        ]
        results = await backend.upload_files(files)
        assert len(results) == 2
        assert all(r.error is None for r in results)
        assert {r.path for r in results} == {"/seed/a.txt", "/seed/b.bin"}

        downloads = await backend.download_files(["/seed/a.txt", "/seed/b.bin"])
        assert all(d.error is None for d in downloads)
        contents = {d.path: d.content for d in downloads}
        assert contents["/seed/a.txt"] == b"alpha"
        assert contents["/seed/b.bin"] == b"\x00\x01\x02\x03"

    async def test_download_partial_failure(self, backend):
        await backend.upload_files([("/exists.txt", b"hi")])
        results = await backend.download_files(["/exists.txt", "/missing.txt"])
        assert len(results) == 2
        ok = next(r for r in results if r.path == "/exists.txt")
        missing = next(r for r in results if r.path == "/missing.txt")
        assert ok.error is None and ok.content == b"hi"
        assert missing.error == "file_not_found"
        assert missing.content is None


# ---------------------------------------------------------------------------
# security
# ---------------------------------------------------------------------------


class TestSecurity:
    async def test_path_traversal_returns_error_code(self, backend):
        result = await backend.read("/../../etc/passwd")
        assert result.error == "permission_denied"
