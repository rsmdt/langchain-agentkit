from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest


def _install_agentfs_stub() -> None:
    try:
        import agentfs_sdk.errors  # noqa: F401
    except ImportError:
        errors = types.ModuleType("agentfs_sdk.errors")

        class ErrnoException(Exception):  # noqa: N818
            def __init__(self, code: str, message: str = "") -> None:
                super().__init__(message or code)
                self.code = code

        errors.ErrnoException = ErrnoException  # type: ignore[attr-defined]
        package = types.ModuleType("agentfs_sdk")
        package.errors = errors  # type: ignore[attr-defined]
        package.__agentkit_test_stub__ = True  # type: ignore[attr-defined]
        sys.modules["agentfs_sdk"] = package
        sys.modules["agentfs_sdk.errors"] = errors


_install_agentfs_stub()

from agentfs_sdk.errors import ErrnoException  # noqa: E402

from langchain_agentkit.backends.agentfs import AgentFSBackend  # noqa: E402


@dataclass
class _Stat:
    directory: bool

    def is_directory(self) -> bool:
        return self.directory

    def is_file(self) -> bool:
        return not self.directory


class _FakeFs:
    def __init__(self, files: dict[str, str], unreadable: frozenset[str] = frozenset()) -> None:
        self.files = dict(files)
        self.unreadable = unreadable

    def _dirs(self) -> set[str]:
        dirs = {"/"}
        for path in self.files:
            parts = path.strip("/").split("/")[:-1]
            for depth in range(1, len(parts) + 1):
                dirs.add("/" + "/".join(parts[:depth]))
        return dirs

    async def stat(self, path: str) -> _Stat:
        if path in self.files:
            return _Stat(directory=False)
        if path in self._dirs():
            return _Stat(directory=True)
        raise ErrnoException("ENOENT", f"no such file: {path}")

    async def readdir(self, path: str) -> list[str]:
        if path in self.unreadable:
            raise ErrnoException("EPERM", "denied")
        prefix = path.rstrip("/") + "/"
        names = {p[len(prefix) :].split("/")[0] for p in self.files if p.startswith(prefix)}
        return list(names)

    async def read_file(self, path: str, encoding: str | None = "utf-8") -> str:
        if path not in self.files:
            raise ErrnoException("ENOENT", f"no such file: {path}")
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


class _FakeAgent:
    def __init__(self, fs: _FakeFs) -> None:
        self.fs = fs


def _backend(files: dict[str, str], unreadable: frozenset[str] = frozenset()) -> AgentFSBackend:
    return AgentFSBackend(_FakeAgent(_FakeFs(files, unreadable)))  # type: ignore[arg-type]


_TREE = {
    "/top.py": "needle\n",
    "/src/a.py": "needle\n",
    "/src/pkg/b.py": "needle\nneedle\n",
    "/lib/d.py": "needle\n",
    "/src/pkg/c.txt": "needle\n",
}


class TestRead:
    async def test_total_lines_and_clamping(self) -> None:
        backend = _backend({"/f.txt": "a\nb\nc"})
        result = await backend.read("/f.txt", offset=-2, limit=1)
        assert result.content == "a\n"
        assert result.total_lines == 3

    async def test_non_positive_limit_is_empty_read(self) -> None:
        result = await _backend({"/f.txt": "a\n"}).read("/f.txt", limit=0)
        assert result.error is None
        assert result.content == ""
        assert result.total_lines == 1


class TestEdit:
    async def test_empty_old_string_rejected(self) -> None:
        fs = _FakeFs({"/f.txt": "abc"})
        backend = AgentFSBackend(_FakeAgent(fs))  # type: ignore[arg-type]
        result = await backend.edit("/f.txt", "", "X", replace_all=True)
        assert result.error is not None
        assert result.error_message
        assert fs.files["/f.txt"] == "abc"


class TestGlob:
    async def test_shared_semantics(self) -> None:
        backend = _backend(_TREE)
        assert (await backend.glob("*.py")).paths == [
            "/lib/d.py",
            "/src/a.py",
            "/src/pkg/b.py",
            "/top.py",
        ]
        assert (await backend.glob("src/**/*.py")).paths == ["/src/a.py", "/src/pkg/b.py"]

    async def test_cap_keeps_earliest(self) -> None:
        result = await _backend(_TREE).glob("*.py", max_results=1)
        assert result.paths == ["/lib/d.py"]
        assert result.truncated is True

    async def test_missing_root_reports_error(self) -> None:
        result = await _backend(_TREE).glob("*", path="/nope")
        assert result.paths == []
        assert result.error_message

    async def test_unreadable_directory_surfaces_error(self) -> None:
        result = await _backend(_TREE, frozenset({"/lib"})).glob("*.py")
        assert "/lib/d.py" not in result.paths
        assert "/top.py" in result.paths
        assert result.error_message


class TestGrep:
    async def test_include_glob_and_cap(self) -> None:
        result = await _backend(_TREE).grep("needle", glob="src/**/*.py", max_matches=2)
        assert [(m["path"], m["line"]) for m in result.matches] == [
            ("/src/a.py", 1),
            ("/src/pkg/b.py", 1),
        ]
        assert result.truncated is True

    @pytest.mark.parametrize("pattern", ["(", "[a-"])
    async def test_invalid_regex_reports_error(self, pattern: str) -> None:
        result = await _backend(_TREE).grep(pattern)
        assert result.matches == []
        assert result.error_message
