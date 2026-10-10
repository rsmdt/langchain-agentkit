from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.backends.search import (
    compile_glob_pattern,
    glob_pattern_regex,
    parse_remote_glob,
    parse_remote_grep,
    remote_glob_command,
    remote_grep_command,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_agentkit.backends.results import GlobResult


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("*.py", "a.py"),
        ("*.py", "a/b/c.py"),
        ("SKILL.md", "skills/x/SKILL.md"),
        ("**/SKILL.md", "SKILL.md"),
        ("**/SKILL.md", "a/b/SKILL.md"),
        ("**/*.py", "a.py"),
        ("src/**/*.py", "src/a.py"),
        ("src/**/*.py", "src/a/b.py"),
        ("src/*.py", "src/a.py"),
        ("src/**", "src/a/b.txt"),
        ("/top.txt", "top.txt"),
        ("file?.txt", "file1.txt"),
        ("[ab].txt", "a.txt"),
        ("[!ab].txt", "c.txt"),
        (".env*", "deep/.env.local"),
    ],
)
def test_matches(pattern: str, path: str) -> None:
    assert compile_glob_pattern(pattern)(path)


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("*.py", "a.pyc"),
        ("*.PY", "a.py"),
        ("src/**/*.py", "lib/a.py"),
        ("src/*.py", "src/a/b.py"),
        ("/top.txt", "nested/top.txt"),
        ("file?.txt", "file12.txt"),
        ("[!ab].txt", "a.txt"),
        ("a.b", "axb"),
    ],
)
def test_rejects(pattern: str, path: str) -> None:
    assert not compile_glob_pattern(pattern)(path)


def test_regex_matches_like_compiled_pattern() -> None:
    assert re.fullmatch(glob_pattern_regex("src/**/[!x]*.py"), "src/a/b.py")


def _tree(root: Path) -> None:
    for rel in ("top.py", "src/a.py", "src/pkg/b.py", "src/pkg/c.txt", "lib/d.py"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"needle in {rel}\n")


def _run(command: str) -> str:
    return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True).stdout


def _virtualizer(root: Path) -> Callable[[str], str]:
    prefix = str(root)

    def to_virtual(path: str) -> str:
        return path.removeprefix(prefix) or "/"

    return to_virtual


def _glob(root: Path, pattern: str, max_results: int = 1000) -> GlobResult:
    output = _run(remote_glob_command(str(root), pattern, max_results))
    return parse_remote_glob(
        output, failure_detail="", cut_short=False, to_virtual=_virtualizer(root)
    )


class TestRemoteSearch:
    def test_glob_uses_shared_semantics(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        assert _glob(tmp_path, "*.py").paths == [
            "/lib/d.py",
            "/src/a.py",
            "/src/pkg/b.py",
            "/top.py",
        ]
        assert _glob(tmp_path, "src/**/*.py").paths == ["/src/a.py", "/src/pkg/b.py"]

    def test_glob_cap_keeps_earliest(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        result = _glob(tmp_path, "*.py", max_results=2)
        assert result.paths == ["/lib/d.py", "/src/a.py"]
        assert result.truncated is True

    def test_glob_missing_root_reports_error(self, tmp_path: Path) -> None:
        result = _glob(tmp_path / "nope", "*")
        assert result.paths == []
        assert result.error_message

    def test_grep_cap_include_and_case(self, tmp_path: Path) -> None:
        _tree(tmp_path)
        (tmp_path / "src" / "many.py").write_text("NEEDLE\n" * 4)
        output = _run(remote_grep_command(str(tmp_path), "needle", "src/*.py", True, 3))
        result = parse_remote_grep(
            output, failure_detail="", cut_short=False, to_virtual=_virtualizer(tmp_path)
        )
        assert [(m["path"], m["line"]) for m in result.matches] == [
            ("/src/a.py", 1),
            ("/src/many.py", 1),
            ("/src/many.py", 2),
        ]
        assert result.matches[0]["text"] == "needle in src/a.py\n"
        assert result.truncated is True

    def test_incomplete_output_without_cut_short_reports_failure(self) -> None:
        result = parse_remote_glob(
            '{"path": "/w/a.py"}\n',
            failure_detail="python3: not found",
            cut_short=False,
            to_virtual=lambda p: p,
        )
        assert result.paths == ["/w/a.py"]
        assert result.error_message is not None
        assert "python3: not found" in result.error_message

    def test_incomplete_output_cut_short_is_truncation(self) -> None:
        result = parse_remote_grep(
            '{"path": "/w/a.py", "line": 1, "text": "x"}\n{"path": "/w/a',
            failure_detail="",
            cut_short=True,
            to_virtual=lambda p: p,
        )
        assert len(result.matches) == 1
        assert result.truncated is True
        assert result.error_message is None
