from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.backends.os import OSBackend

if TYPE_CHECKING:
    from pathlib import Path

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_until_gone(pid: int, seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


class TestExecuteProcessGroup:
    @posix_only
    async def test_timeout_kills_background_grandchild(self, tmp_path: Path) -> None:
        backend = OSBackend(str(tmp_path))
        result = await backend.execute("sleep 30 & echo $!; sleep 30", timeout=1)
        assert result["exit_code"] == -1
        grandchild = int(result["output"].split()[0])
        try:
            assert await _wait_until_gone(grandchild)
        finally:
            if _alive(grandchild):
                os.kill(grandchild, 9)

    async def test_stdin_reads_eof_immediately(self, tmp_path: Path) -> None:
        backend = OSBackend(str(tmp_path))
        read_end, write_end = os.pipe()
        saved_stdin = os.dup(0)
        os.dup2(read_end, 0)
        try:
            result = await backend.execute("cat; echo done", timeout=3)
        finally:
            os.dup2(saved_stdin, 0)
            for fd in (saved_stdin, read_end, write_end):
                os.close(fd)
        assert result["exit_code"] == 0
        assert result["output"].strip() == "done"


class TestReadWindow:
    async def test_reports_total_lines(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\nc\nd\n")
        result = await OSBackend(str(tmp_path)).read("/f.txt", offset=1, limit=2)
        assert result.content == "b\nc\n"
        assert result.total_lines == 4

    async def test_total_lines_counts_unterminated_last_line(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb")
        result = await OSBackend(str(tmp_path)).read("/f.txt")
        assert result.total_lines == 2

    async def test_negative_offset_clamps_to_start(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("a\nb\n")
        result = await OSBackend(str(tmp_path)).read("/f.txt", offset=-5, limit=1)
        assert result.error is None
        assert result.content == "a\n"

    @pytest.mark.parametrize("limit", [0, -3])
    async def test_non_positive_limit_is_empty_read(self, tmp_path: Path, limit: int) -> None:
        (tmp_path / "f.txt").write_text("a\nb\n")
        result = await OSBackend(str(tmp_path)).read("/f.txt", limit=limit)
        assert result.error is None
        assert result.content == ""
        assert result.total_lines == 2


class TestEditEmptyOldString:
    async def test_rejected_without_touching_file(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("abc")
        result = await OSBackend(str(tmp_path)).edit("/f.txt", "", "X", replace_all=True)
        assert result.error is not None
        assert result.error_message
        assert (tmp_path / "f.txt").read_text() == "abc"


def _tree(root: Path) -> None:
    for rel in ("top.py", "src/a.py", "src/pkg/b.py", "src/pkg/c.txt", "lib/d.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"needle in {rel}\n")


class TestGlobSemantics:
    async def test_bare_pattern_matches_at_any_depth(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).glob("*.py")
        assert result.paths == ["/lib/d.py", "/src/a.py", "/src/pkg/b.py", "/top.py"]
        assert result.truncated is False
        assert result.error_message is None

    async def test_globstar_matches_zero_directories(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).glob("src/**/*.py")
        assert result.paths == ["/src/a.py", "/src/pkg/b.py"]

    async def test_slash_pattern_is_relative_to_search_root(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).glob("pkg/*.py", path="/src")
        assert result.paths == ["/src/pkg/b.py"]

    async def test_cap_keeps_earliest_and_flags_truncation(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).glob("*.py", max_results=2)
        assert result.paths == ["/lib/d.py", "/src/a.py"]
        assert result.truncated is True

    async def test_exact_cap_is_not_truncated(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).glob("*.py", max_results=4)
        assert len(result.paths) == 4
        assert result.truncated is False

    async def test_escaping_path_reports_error(self, tmp_path: Path) -> None:
        result = await OSBackend(str(tmp_path)).glob("*", path="/../..")
        assert result.paths == []
        assert result.error_message

    async def test_missing_path_reports_error(self, tmp_path: Path) -> None:
        result = await OSBackend(str(tmp_path)).glob("*", path="/nope")
        assert result.paths == []
        assert result.error_message

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
    async def test_unreadable_directory_surfaces_error_with_partial_results(
        self, tmp_path: Path
    ) -> None:
        _tree(tmp_path)
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "hidden.py").write_text("x")
        locked.chmod(0)
        try:
            result = await OSBackend(str(tmp_path)).glob("*.py")
        finally:
            locked.chmod(0o755)
        assert "/top.py" in result.paths
        assert result.error_message


class TestGrep:
    async def test_include_glob_uses_shared_semantics(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = await OSBackend(str(tmp_path)).grep("needle", glob="src/**/*.py")
        assert [m["path"] for m in result.matches] == ["/src/a.py", "/src/pkg/b.py"]

    async def test_cap_keeps_earliest_and_flags_truncation(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("hit\n" * 5)
        result = await OSBackend(str(tmp_path)).grep("hit", max_matches=3)
        assert [m["line"] for m in result.matches] == [1, 2, 3]
        assert result.truncated is True

    async def test_invalid_regex_reports_error(self, tmp_path: Path) -> None:
        result = await OSBackend(str(tmp_path)).grep("(")
        assert result.matches == []
        assert result.error_message

    async def test_escaping_path_reports_error(self, tmp_path: Path) -> None:
        result = await OSBackend(str(tmp_path)).grep("x", path="/../..")
        assert result.matches == []
        assert result.error_message

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
    async def test_unreadable_file_surfaces_error(self, tmp_path: Path) -> None:
        (tmp_path / "ok.txt").write_text("needle\n")
        locked = tmp_path / "locked.txt"
        locked.write_text("needle\n")
        locked.chmod(0)
        try:
            result = await OSBackend(str(tmp_path)).grep("needle")
        finally:
            locked.chmod(0o644)
        assert [m["path"] for m in result.matches] == ["/ok.txt"]
        assert result.error_message
