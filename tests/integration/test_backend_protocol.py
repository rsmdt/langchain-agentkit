"""FilesystemProtocol conformance tests — run against every backend.

These tests verify the contract of both capability protocols:
``FilesystemProtocol`` (file ops + bulk transfer, required) and
``SandboxProtocol`` (adds execute/environment, optional). Backends
that don't claim ``SandboxProtocol`` skip the sandbox tests via the
``sandbox_backend`` capability fixture.

The ``backend`` fixture is parameterized so the same tests run against
every available backend:

- ``os`` — always available (OSBackend with temp directory)
- ``daytona`` — only when DAYTONA_API_URL is set and SDK is installed
- ``agentfs`` — only when the agentfs-sdk is installed (no env vars
  required; uses a temp .db file)
- ``mirage`` — only when the mirage-ai SDK is installed (no env vars
  required; uses an in-memory ``RAMResource`` mounted at ``/``)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from langchain_agentkit.backends import (
    FilesystemProtocol,
    SandboxProtocol,
)
from langchain_agentkit.backends.os import OSBackend

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
    # The unit suite registers a stub `daytona_sdk` module; real conformance
    # must only run against a genuine install.
    try:
        import daytona_sdk

        if getattr(daytona_sdk, "__agentkit_test_stub__", False):
            return False
    except ImportError:
        pass
    cls, _ = _import_daytona()
    return cls is not None


# ---------------------------------------------------------------------------
# AgentFS helpers
# ---------------------------------------------------------------------------


def _agentfs_available() -> bool:
    """AgentFS conformance runs whenever agentfs-sdk is importable.

    No env vars or external services required — AgentFS is a local
    SQLite-backed virtual FS, and the fixture spins up a temp .db.
    """
    try:
        import agentfs_sdk
    except ImportError:
        return False
    return not getattr(agentfs_sdk, "__agentkit_test_stub__", False)


# ---------------------------------------------------------------------------
# Bubblewrap helpers
# ---------------------------------------------------------------------------


def _bubblewrap_available() -> bool:
    """Bubblewrap conformance runs only on Linux with bwrap installed.

    No SDK or env vars required. The CI/dev path is to invoke this
    suite inside ``tests/docker/Dockerfile`` so the matrix executes
    on a host where ``bwrap`` is real.
    """
    return sys.platform == "linux" and shutil.which("bwrap") is not None


# ---------------------------------------------------------------------------
# Mirage helpers
# ---------------------------------------------------------------------------


def _mirage_available() -> bool:
    """Mirage conformance runs whenever the mirage-ai SDK is importable.

    No env vars or external services required — the fixture mounts an
    in-memory ``RAMResource`` at ``/``.
    """
    try:
        import mirage
    except ImportError:
        return False
    # The unit suite registers a stub `mirage` module; real conformance must
    # only run against a genuine mirage-ai install.
    return not getattr(mirage, "__agentkit_test_stub__", False)


# ---------------------------------------------------------------------------
# Parameterized backend fixture
# ---------------------------------------------------------------------------

_BACKEND_IDS = ["os"]
if _daytona_available():
    _BACKEND_IDS.append("daytona")
if _agentfs_available():
    _BACKEND_IDS.append("agentfs")
if _bubblewrap_available():
    _BACKEND_IDS.append("bubblewrap")
if _mirage_available():
    _BACKEND_IDS.append("mirage")


@pytest.fixture(params=_BACKEND_IDS)
async def backend(request):
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

    elif request.param == "agentfs":
        from agentfs_sdk import AgentFS, AgentFSOptions

        from langchain_agentkit.backends.agentfs import AgentFSBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "agent.db")
            agent = await AgentFS.open(AgentFSOptions(path=db_path))
            try:
                yield AgentFSBackend(agent)
            finally:
                await agent.close()

    elif request.param == "bubblewrap":
        from langchain_agentkit.backends.bubblewrap import BubblewrapBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            yield BubblewrapBackend(tmpdir)

    elif request.param == "mirage":
        from mirage import MountMode, RAMResource, Workspace

        from langchain_agentkit.backends.mirage import MirageBackend

        # Mount the RAM resource WRITE so the conformance suite (which
        # writes test fixtures into / before reading them back) can
        # actually persist data. The default ``MountMode.READ`` would
        # surface ``permission_denied`` on every write.
        ws = Workspace({"/": (RAMResource(), MountMode.WRITE)})
        try:
            yield MirageBackend(ws)
        finally:
            await ws.close()


# ---------------------------------------------------------------------------
# Capability fixture — skip sandbox tests when the parametrized backend
# doesn't claim ``SandboxProtocol``. Lets filesystem-only backends (e.g.
# AgentFS) coexist in the matrix without spurious failures.
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox_backend(backend):
    if not isinstance(backend, SandboxProtocol):
        pytest.skip(
            f"{type(backend).__name__} does not implement SandboxProtocol (execute / environment)"
        )
    return backend


# ---------------------------------------------------------------------------
# Protocol tier conformance
# ---------------------------------------------------------------------------


class TestProtocolTiers:
    def test_implements_filesystem_protocol(self, backend):
        # Every backend in the matrix must satisfy the file tier
        # (read/write/edit/glob/grep + upload/download).
        assert isinstance(backend, FilesystemProtocol)

    def test_implements_sandbox_protocol(self, sandbox_backend):
        # Skipped via the capability fixture for filesystem-only backends.
        assert isinstance(sandbox_backend, SandboxProtocol)


# ---------------------------------------------------------------------------
# read / read_bytes
# ---------------------------------------------------------------------------


class TestRead:
    async def test_reports_total_lines(self, backend):
        await backend.write("/test.txt", "a\nb\nc\nd")
        result = await backend.read("/test.txt", offset=1, limit=2)
        assert result.content == "b\nc\n"
        assert result.total_lines == 4

    async def test_negative_offset_clamps_to_start(self, backend):
        await backend.write("/test.txt", "a\nb\n")
        result = await backend.read("/test.txt", offset=-3, limit=1)
        assert result.error is None
        assert result.content == "a\n"

    async def test_non_positive_limit_is_empty_read(self, backend):
        await backend.write("/test.txt", "a\nb\n")
        result = await backend.read("/test.txt", limit=0)
        assert result.error is None
        assert result.content == ""
        assert result.total_lines == 2

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
    async def test_empty_old_string_rejected(self, backend):
        await backend.write("/test.txt", "abc")
        result = await backend.edit("/test.txt", "", "X", replace_all=True)
        assert result.error is not None
        assert result.error_message
        assert (await backend.read("/test.txt")).content == "abc"

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
        result = await backend.glob("**/*.py", path="/src")
        assert result.paths == ["/src/a.py", "/src/b.py"]

    async def test_no_matches(self, backend):
        result = await backend.glob("*.xyz")
        assert result.paths == []
        assert result.truncated is False

    async def test_bare_pattern_matches_at_any_depth(self, backend):
        await backend.write("/g/top.py", "t")
        await backend.write("/g/src/pkg/deep.py", "d")
        result = await backend.glob("*.py", path="/g")
        assert result.paths == ["/g/src/pkg/deep.py", "/g/top.py"]

    async def test_globstar_spans_zero_directories(self, backend):
        await backend.write("/g/src/a.py", "a")
        await backend.write("/g/src/pkg/b.py", "b")
        await backend.write("/g/lib/c.py", "c")
        result = await backend.glob("src/**/*.py", path="/g")
        assert result.paths == ["/g/src/a.py", "/g/src/pkg/b.py"]

    async def test_cap_keeps_earliest_results(self, backend):
        for name in ("a", "b", "c"):
            await backend.write(f"/g/{name}.txt", name)
        result = await backend.glob("*.txt", path="/g", max_results=2)
        assert result.paths == ["/g/a.txt", "/g/b.txt"]
        assert result.truncated is True

    async def test_missing_directory_reports_error(self, backend):
        result = await backend.glob("*", path="/does-not-exist")
        assert result.paths == []
        assert result.error_message


# ---------------------------------------------------------------------------
# glob — recursive-basename patterns on Mirage sub-path mounts
# ---------------------------------------------------------------------------
#
# Mirage-specific regression: ``**/<basename>`` patterns
# (``**/SKILL.md``, ``**/*.md``) on a sub-path mount must find files at
# every depth — skill discovery depends on it. Mirage lists files with a
# plain ``find -type f`` and matches them with the shared glob matcher.
#
# These cases need a sub-path mount to reproduce, so they run only
# against a dedicated Mirage workspace — not the shared backend matrix
# (which mounts at ``/``).


@pytest.fixture
async def mirage_subpath_backend():
    """Yield a ``MirageBackend`` with a sub-path RAM mount + matching workdir.

    The bug only manifests when the resolved workspace path is a
    sub-path of the workspace mount root, because Mirage's ``find -path``
    pattern evaluation differs subtly from GNU find at the mount boundary.
    """
    if not _mirage_available():
        pytest.skip("mirage-ai SDK not installed")

    from mirage import MountMode, RAMResource, Workspace

    from langchain_agentkit.backends.mirage import MirageBackend

    ws = Workspace({"/workspace": (RAMResource(), MountMode.WRITE)})
    try:
        yield MirageBackend(ws, workdir="/workspace")
    finally:
        await ws.close()


@pytest.fixture
async def mirage_root_backend():
    """Yield a ``MirageBackend`` mounted at ``/`` (regression baseline)."""
    if not _mirage_available():
        pytest.skip("mirage-ai SDK not installed")

    from mirage import MountMode, RAMResource, Workspace

    from langchain_agentkit.backends.mirage import MirageBackend

    ws = Workspace({"/": (RAMResource(), MountMode.WRITE)})
    try:
        yield MirageBackend(ws, workdir="/")
    finally:
        await ws.close()


class TestGlobRecursiveBasename:
    async def test_subpath_mount_deeply_nested_basename(self, mirage_subpath_backend):
        # (a) ``**/SKILL.md`` on a sub-path mount with a deeply-nested
        # SKILL.md must return it. This is the exact scenario that
        # broke SkillsExtension discovery before the fix.
        await mirage_subpath_backend.write("/skills/foo/SKILL.md", "foo skill")
        await mirage_subpath_backend.write("/skills/bar/baz/SKILL.md", "bar skill")
        await mirage_subpath_backend.write("/skills/quux/notes.md", "not a skill")
        result = await mirage_subpath_backend.glob("**/SKILL.md")
        assert sorted(result.paths) == [
            "/skills/bar/baz/SKILL.md",
            "/skills/foo/SKILL.md",
        ]

    async def test_subpath_mount_flat_depth_extension(self, mirage_subpath_backend):
        # (b) ``**/*.md`` at a non-root mount must surface flat
        # depth-0 files too. Pre-fix, the ``-path`` translation
        # required at least one intermediate directory and missed them.
        await mirage_subpath_backend.write("/top.md", "top-level note")
        await mirage_subpath_backend.write("/notes/deep/extra.md", "deep note")
        result = await mirage_subpath_backend.glob("**/*.md")
        assert sorted(result.paths) == ["/notes/deep/extra.md", "/top.md"]

    async def test_root_mount_basename_regression(self, mirage_root_backend):
        # (c) Root-mount regression: ``**/SKILL.md`` keeps working when
        # the workspace is mounted at ``/`` (the pre-fix happy path).
        await mirage_root_backend.write("/skills/foo/SKILL.md", "x")
        await mirage_root_backend.write("/skills/bar/SKILL.md", "y")
        result = await mirage_root_backend.glob("**/SKILL.md")
        assert sorted(result.paths) == ["/skills/bar/SKILL.md", "/skills/foo/SKILL.md"]


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    async def test_finds_pattern(self, backend):
        await backend.write("/test.txt", "line one\nline two\nline three")
        result = await backend.grep("two", path="/")
        assert ("/test.txt", 2) in [(m["path"], m["line"]) for m in result.matches]

    async def test_ignore_case(self, backend):
        await backend.write("/f.txt", "Hello World\ngoodbye")
        result = await backend.grep("hello", ignore_case=True)
        assert len(result.matches) == 1
        assert "Hello World" in result.matches[0]["text"]

    async def test_case_sensitive_by_default(self, backend):
        await backend.write("/f.txt", "Hello World")
        result = await backend.grep("hello")
        assert result.matches == []

    async def test_no_matches(self, backend):
        await backend.write("/f.txt", "nothing here")
        result = await backend.grep("missing")
        assert result.matches == []
        assert result.truncated is False

    async def test_include_glob_uses_shared_semantics(self, backend):
        await backend.write("/g/src/a.py", "hit")
        await backend.write("/g/src/pkg/b.py", "hit")
        await backend.write("/g/src/c.txt", "hit")
        await backend.write("/g/lib/d.py", "hit")
        result = await backend.grep("hit", path="/g", glob="src/**/*.py")
        assert sorted(m["path"] for m in result.matches) == ["/g/src/a.py", "/g/src/pkg/b.py"]
        result = await backend.grep("hit", path="/g", glob="*.py")
        assert len(result.matches) == 3

    async def test_cap_keeps_earliest_matches(self, backend):
        await backend.write("/g/f.txt", "hit\n" * 5)
        result = await backend.grep("hit", path="/g", max_matches=2)
        assert [m["line"] for m in result.matches] == [1, 2]
        assert result.truncated is True


# ---------------------------------------------------------------------------
# execute (SandboxProtocol)
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_echo(self, sandbox_backend):
        result = await sandbox_backend.execute("echo hello")
        assert result["exit_code"] == 0
        assert "hello" in result["output"]

    async def test_nonzero_exit(self, sandbox_backend):
        # ``false`` is portable across real shells and MirageBackend's
        # curated command set; ``exit 1`` is a bash builtin that Mirage's
        # in-process shell doesn't dispatch.
        result = await sandbox_backend.execute("false")
        assert result["exit_code"] == 1


# ---------------------------------------------------------------------------
# environment (SandboxProtocol)
# ---------------------------------------------------------------------------


class TestEnvironment:
    async def test_returns_environment_snapshot(self, sandbox_backend):
        env = await sandbox_backend.environment()
        # os encodes uname -srm shape: kernel-name + release + arch.
        # The leading word identifies the platform family unambiguously.
        assert env.os, "os must be a non-empty string"
        assert env.os.split()[0] in {"Linux", "Darwin", "Windows"}
        # cwd reflects backend root / workdir
        assert env.cwd
        # shell is a non-empty string. Real-shell backends report the
        # POSIX path (``/bin/bash`` etc.); virtualized shells like
        # MirageBackend report a synthetic identifier (``mirage-bash``).
        assert env.shell
        # available_tools is a frozenset of tool names. The size and
        # contents depend on the backend's probing strategy:
        # OS/Daytona/Bubblewrap/AgentFS probe against ``PROBED_TOOLS``
        # (so the result is a subset); MirageBackend exposes its full
        # in-process command registry (typically dozens of names plus
        # any resource-specific commands). The protocol contract is
        # just "frozenset of strings the LLM can rely on."
        assert isinstance(env.available_tools, frozenset)
        assert all(isinstance(t, str) and t for t in env.available_tools)

    async def test_cached(self, sandbox_backend):
        env1 = await sandbox_backend.environment()
        env2 = await sandbox_backend.environment()
        assert env1 is env2  # same object — backend caches


# ---------------------------------------------------------------------------
# upload / download (FilesystemProtocol)
# ---------------------------------------------------------------------------


class TestFileTransfer:
    async def test_upload_roundtrip(self, backend):
        files = [
            ("/seed/a.txt", b"alpha"),
            ("/seed/b.bin", b"\x00\x01\x02\x03"),
        ]
        results = await backend.upload(files)
        assert len(results) == 2
        assert all(r.error is None for r in results)
        assert {r.path for r in results} == {"/seed/a.txt", "/seed/b.bin"}

        downloads = await backend.download(["/seed/a.txt", "/seed/b.bin"])
        assert all(d.error is None for d in downloads)
        contents = {d.path: d.content for d in downloads}
        assert contents["/seed/a.txt"] == b"alpha"
        assert contents["/seed/b.bin"] == b"\x00\x01\x02\x03"

    async def test_download_partial_failure(self, backend):
        await backend.upload([("/exists.txt", b"hi")])
        results = await backend.download(["/exists.txt", "/missing.txt"])
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
