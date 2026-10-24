"""Unit tests for :class:`MirageBackend` — path resolution, error mapping.

Behavior the integration matrix can't reach cleanly:

- ``_resolve()`` edge cases for both workdir styles (``"/"`` and a
  sub-path namespace).
- Exception → ``FileError`` translation table (``_map_exc``).
- Constructor invariants (workdir normalization).
- ``glob()`` command-translation branches — verified by intercepting
  the emitted shell command so we cover the ``find -name`` vs
  ``find -path`` fork without booting a real workspace.

Protocol-level conformance is exercised in
``tests/integration/test_backend_protocol.py`` against a real
``RAMVFS`` mount.
"""

from __future__ import annotations

import pytest

from langchain_agentkit.backends.mirage import MirageBackend, _map_exc

# ---------------------------------------------------------------------------
# Constructor invariants
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_workdir_root_normalized(self) -> None:
        backend = MirageBackend(workspace=_FakeWorkspace(), workdir="/")
        assert backend.workdir == "/"

    def test_workdir_trailing_slash_stripped(self) -> None:
        backend = MirageBackend(workspace=_FakeWorkspace(), workdir="/workspace/")
        assert backend.workdir == "/workspace"

    def test_workdir_double_trailing_slash_stripped(self) -> None:
        backend = MirageBackend(workspace=_FakeWorkspace(), workdir="/workspace//")
        assert backend.workdir == "/workspace"

    def test_workspace_property_returns_handle(self) -> None:
        ws = _FakeWorkspace()
        backend = MirageBackend(workspace=ws)
        assert backend.workspace is ws


# ---------------------------------------------------------------------------
# Path resolution — workdir = "/"
# ---------------------------------------------------------------------------


class TestResolveRootWorkdir:
    @pytest.fixture
    def backend(self) -> MirageBackend:
        return MirageBackend(workspace=_FakeWorkspace(), workdir="/")

    def test_root_path_returns_root(self, backend: MirageBackend) -> None:
        assert backend._resolve("/") == "/"

    def test_empty_path_returns_workdir(self, backend: MirageBackend) -> None:
        assert backend._resolve("") == "/"

    def test_absolute_path_passes_through(self, backend: MirageBackend) -> None:
        assert backend._resolve("/data/notes.md") == "/data/notes.md"

    def test_relative_path_resolved_under_root(self, backend: MirageBackend) -> None:
        # A bare "foo" is treated as "/foo" — same as DaytonaBackend.
        assert backend._resolve("foo") == "/foo"

    def test_double_slash_normalized(self, backend: MirageBackend) -> None:
        assert backend._resolve("/data//notes.md") == "/data/notes.md"

    def test_dot_segments_normalized(self, backend: MirageBackend) -> None:
        assert backend._resolve("/data/./notes.md") == "/data/notes.md"

    def test_dotdot_within_tree_collapses(self, backend: MirageBackend) -> None:
        # /a/b/../c collapses to /a/c — that's normal navigation, not an escape.
        assert backend._resolve("/a/b/../c") == "/a/c"

    def test_traversal_above_root_blocked(self, backend: MirageBackend) -> None:
        # Even with workdir="/" we reject paths whose normalized form
        # has leading ".." segments — that's the only way an absolute
        # path can express "above the virtual root."
        with pytest.raises(PermissionError):
            backend._resolve("/../../etc/passwd")

    def test_relative_traversal_blocked(self, backend: MirageBackend) -> None:
        with pytest.raises(PermissionError):
            backend._resolve("../etc/passwd")


# ---------------------------------------------------------------------------
# Path resolution — workdir = "/workspace"
# ---------------------------------------------------------------------------


class TestResolveSubWorkdir:
    @pytest.fixture
    def backend(self) -> MirageBackend:
        return MirageBackend(workspace=_FakeWorkspace(), workdir="/workspace")

    def test_root_path_returns_workdir(self, backend: MirageBackend) -> None:
        assert backend._resolve("/") == "/workspace"

    def test_empty_path_returns_workdir(self, backend: MirageBackend) -> None:
        assert backend._resolve("") == "/workspace"

    def test_absolute_under_workdir_passes_through(self, backend: MirageBackend) -> None:
        assert backend._resolve("/workspace/notes.md") == "/workspace/notes.md"

    def test_workdir_relative_absolute_resolved_under_workdir(self, backend: MirageBackend) -> None:
        # "/notes.md" with workdir="/workspace" is treated as
        # workdir-relative — matches DaytonaBackend's documented
        # shorthand for paths the LLM emits with a leading slash.
        assert backend._resolve("/notes.md") == "/workspace/notes.md"

    def test_relative_path_resolved_under_workdir(self, backend: MirageBackend) -> None:
        assert backend._resolve("notes.md") == "/workspace/notes.md"

    def test_dotdot_within_workdir_collapses(self, backend: MirageBackend) -> None:
        assert backend._resolve("/workspace/a/../b") == "/workspace/b"

    def test_traversal_above_workdir_blocked(self, backend: MirageBackend) -> None:
        with pytest.raises(PermissionError):
            backend._resolve("/../../etc/passwd")

    def test_workdir_relative_traversal_blocked(self, backend: MirageBackend) -> None:
        with pytest.raises(PermissionError):
            backend._resolve("../etc/passwd")


# ---------------------------------------------------------------------------
# Exception → FileError mapping
# ---------------------------------------------------------------------------


class TestMapExc:
    def test_file_not_found(self) -> None:
        assert _map_exc(FileNotFoundError("missing")) == "file_not_found"

    def test_is_a_directory(self) -> None:
        assert _map_exc(IsADirectoryError("dir")) == "is_directory"

    def test_not_a_directory(self) -> None:
        assert _map_exc(NotADirectoryError("file used as dir")) == "invalid_path"

    def test_permission_denied(self) -> None:
        assert _map_exc(PermissionError("no access")) == "permission_denied"

    def test_unicode_decode_error(self) -> None:
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        assert _map_exc(exc) == "decode_error"

    def test_generic_oserror_falls_through_to_io_error(self) -> None:
        # OSError without one of the more specific subclasses → io_error.
        assert _map_exc(OSError("disk failure")) == "io_error"

    def test_unknown_exception_falls_through_to_io_error(self) -> None:
        assert _map_exc(RuntimeError("weird")) == "io_error"


_LISTING = "\n".join(
    [
        "/workspace/top.md",
        "/workspace/skills/foo/SKILL.md",
        "/workspace/skills/bar/baz/SKILL.md",
        "/workspace/skills/quux/notes.md",
        "/workspace/src/a.py",
    ]
)


class TestGlob:
    async def test_lists_with_plain_find(self) -> None:
        ws = _ListingWorkspace(_LISTING)
        await MirageBackend(workspace=ws, workdir="/workspace").glob("**/SKILL.md")
        assert ws.last_cmd == "find '/workspace' -type f"

    async def test_recursive_basename(self) -> None:
        backend = MirageBackend(workspace=_ListingWorkspace(_LISTING), workdir="/workspace")
        result = await backend.glob("**/SKILL.md")
        assert result.paths == ["/skills/bar/baz/SKILL.md", "/skills/foo/SKILL.md"]

    async def test_bare_pattern_matches_any_depth(self) -> None:
        backend = MirageBackend(workspace=_ListingWorkspace(_LISTING), workdir="/workspace")
        result = await backend.glob("*.md")
        assert result.paths == [
            "/skills/bar/baz/SKILL.md",
            "/skills/foo/SKILL.md",
            "/skills/quux/notes.md",
            "/top.md",
        ]

    async def test_compound_pattern_relative_to_search_root(self) -> None:
        backend = MirageBackend(workspace=_ListingWorkspace(_LISTING), workdir="/workspace")
        result = await backend.glob("**/foo/*.md", path="/skills")
        assert result.paths == ["/skills/foo/SKILL.md"]

    async def test_cap_keeps_earliest(self) -> None:
        backend = MirageBackend(workspace=_ListingWorkspace(_LISTING), workdir="/workspace")
        result = await backend.glob("*.md", max_results=1)
        assert result.paths == ["/skills/bar/baz/SKILL.md"]
        assert result.truncated is True

    async def test_find_failure_surfaces_error(self) -> None:
        ws = _ListingWorkspace("", stderr="find: /workspace/nope: No such file", exit_code=1)
        result = await MirageBackend(workspace=ws, workdir="/workspace").glob("*", path="/nope")
        assert result.paths == []
        assert result.error_message == "find: /workspace/nope: No such file"

    async def test_traversal_reports_error(self) -> None:
        ws = _ListingWorkspace(_LISTING)
        result = await MirageBackend(workspace=ws, workdir="/workspace").glob("*", path="/../..")
        assert result.paths == []
        assert result.error_message
        assert ws.last_cmd is None


class TestGrep:
    async def test_include_glob_cap_and_case(self) -> None:
        ws = _ListingWorkspace(
            _LISTING,
            files={
                "/workspace/src/a.py": b"Hit\nmiss\nhit\n",
                "/workspace/top.md": b"hit\n",
            },
        )
        backend = MirageBackend(workspace=ws, workdir="/workspace")
        result = await backend.grep("hit", glob="src/*.py", ignore_case=True, max_matches=1)
        assert result.matches == [{"path": "/src/a.py", "line": 1, "text": "Hit\n"}]
        assert result.truncated is True

    async def test_unreadable_file_surfaces_error(self) -> None:
        ws = _ListingWorkspace(
            "/workspace/a.txt\n/workspace/b.txt", files={"/workspace/b.txt": b"x\n"}
        )
        result = await MirageBackend(workspace=ws, workdir="/workspace").grep("x")
        assert [m["path"] for m in result.matches] == ["/b.txt"]
        assert result.error_message


class TestReadAndEdit:
    async def test_read_total_lines_and_window_clamping(self) -> None:
        ws = _ListingWorkspace("", files={"/f.txt": b"a\nb\nc"})
        backend = MirageBackend(workspace=ws)
        result = await backend.read("/f.txt", offset=-1, limit=0)
        assert result.content == ""
        assert result.total_lines == 3

    async def test_edit_rejects_empty_old_string(self) -> None:
        ws = _ListingWorkspace("", files={"/f.txt": b"abc"})
        result = await MirageBackend(workspace=ws).edit("/f.txt", "", "X")
        assert result.error is not None
        assert ws.files["/f.txt"] == b"abc"


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    """Stand-in for :class:`mirage.Workspace`.

    These tests exercise pure-Python path/error logic that doesn't touch
    the Mirage SDK — a real workspace would slow tests down without
    adding coverage. Integration tests in
    ``tests/integration/test_backend_protocol.py`` validate the SDK-bound
    paths against an actual ``RAMVFS`` mount.
    """


class _ListingWorkspace:
    def __init__(
        self,
        stdout: str,
        *,
        stderr: str = "",
        exit_code: int = 0,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.last_cmd: str | None = None
        self.files = dict(files or {})
        self._io = _StaticIO(stdout, stderr, exit_code)
        self.vfs = _FakeVFS(self.files)

    async def shell(self, cmd: str) -> _StaticIO:
        self.last_cmd = cmd
        return self._io


class _StaticIO:
    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.exit_code = exit_code

    async def stdout_str(self) -> str:
        return self._stdout

    async def stderr_str(self) -> str:
        return self._stderr


class _FakeVFS:
    def __init__(self, files: dict[str, bytes]) -> None:
        self._files = files

    async def read(self, path: str, raw: bool = False) -> bytes:
        if path not in self._files:
            raise PermissionError(f"denied: {path}")
        return self._files[path]
