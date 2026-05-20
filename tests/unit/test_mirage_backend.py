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
``RAMResource`` mount.
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


# ---------------------------------------------------------------------------
# glob() command translation — branch coverage without booting a workspace
# ---------------------------------------------------------------------------


class TestGlobCommandTranslation:
    """Assert which ``find`` flavor each pattern shape emits.

    Behavioral coverage of the recursive-basename fix lives in
    ``tests/integration/test_backend_protocol.py::TestGlobRecursiveBasename``
    (which needs a real ``RAMResource`` to exercise ``find -path`` /
    ``-name`` evaluation under a sub-path mount). These tests cover the
    branch fork: the regex short-circuit must take ``**/<basename>``
    patterns to ``-name`` and leave compound recursive patterns
    (``**/skills/*.md``) on the ``-path`` branch.
    """

    async def test_basename_recursive_uses_find_name(self) -> None:
        ws = _CommandCapturingWorkspace()
        backend = MirageBackend(workspace=ws, workdir="/workspace")
        await backend.glob("**/SKILL.md")
        assert ws.last_cmd is not None
        assert "-name 'SKILL.md'" in ws.last_cmd
        assert "-path" not in ws.last_cmd

    async def test_basename_glob_extension_uses_find_name(self) -> None:
        ws = _CommandCapturingWorkspace()
        backend = MirageBackend(workspace=ws, workdir="/workspace")
        await backend.glob("**/*.md")
        assert ws.last_cmd is not None
        assert "-name '*.md'" in ws.last_cmd
        assert "-path" not in ws.last_cmd

    async def test_compound_recursive_keeps_find_path(self) -> None:
        # (d) Regression: compound recursive patterns must keep the
        # existing ``-path`` translation rather than being miscaptured
        # by the new ``**/<basename>`` short-circuit.
        ws = _CommandCapturingWorkspace()
        backend = MirageBackend(workspace=ws, workdir="/workspace")
        await backend.glob("**/skills/*.md")
        assert ws.last_cmd is not None
        assert "-path '/workspace/*/skills/*.md'" in ws.last_cmd
        assert "-name" not in ws.last_cmd

    async def test_flat_pattern_uses_find_name(self) -> None:
        ws = _CommandCapturingWorkspace()
        backend = MirageBackend(workspace=ws, workdir="/workspace")
        await backend.glob("*.py")
        assert ws.last_cmd is not None
        assert "-name '*.py'" in ws.last_cmd
        assert "-path" not in ws.last_cmd


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    """Stand-in for :class:`mirage.Workspace`.

    These tests exercise pure-Python path/error logic that doesn't touch
    the Mirage SDK — a real workspace would slow tests down without
    adding coverage. Integration tests in
    ``tests/integration/test_backend_protocol.py`` validate the SDK-bound
    paths against an actual ``RAMResource`` mount.
    """


class _CommandCapturingWorkspace:
    """Stand-in for :class:`mirage.Workspace` that records ``execute`` calls.

    Returns an empty stdout so ``glob()`` short-circuits to ``[]`` — we
    only care about the emitted command string, not the result. Keeps
    these unit tests fast and SDK-free while still exercising the
    branch fork in :meth:`MirageBackend.glob`.
    """

    def __init__(self) -> None:
        self.last_cmd: str | None = None

    async def execute(self, cmd: str) -> _EmptyIO:
        self.last_cmd = cmd
        return _EmptyIO()


class _EmptyIO:
    async def stdout_str(self) -> str:
        return ""
