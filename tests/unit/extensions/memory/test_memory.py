"""Tests for MemoryExtension."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.extensions.memory import MemoryExtension
from langchain_agentkit.extensions.memory.extension import sanitize_path

if TYPE_CHECKING:
    from pathlib import Path

# --- sanitize_path ---


def test_sanitize_path_replaces_non_alphanumerics() -> None:
    assert sanitize_path("/Users/foo/project") == "-Users-foo-project"


def test_sanitize_path_preserves_alphanumerics() -> None:
    assert sanitize_path("abc123XYZ") == "abc123XYZ"


def test_sanitize_path_truncates_with_hash_suffix() -> None:
    name = "a" * 250
    result = sanitize_path(name, max_length=200)
    expected_hash = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    assert result == "a" * 200 + "-" + expected_hash
    assert len(result) == 200 + 1 + 12


def test_sanitize_path_equal_to_max_length_not_truncated() -> None:
    name = "a" * 200
    assert sanitize_path(name, max_length=200) == name


# --- Extension defaults / no I/O in init ---


def test_init_no_filesystem_io(tmp_path: Path) -> None:
    # Even with a non-existent path, construction must succeed.
    ext = MemoryExtension(path=tmp_path / "nonexistent", project_discovery=False)
    assert ext.tools == []
    assert ext.state_schema is None


def test_prompt_returns_none_when_no_file(tmp_path: Path) -> None:
    ext = MemoryExtension(path=tmp_path, project_discovery=False)
    assert ext.prompt({}, None) is None


# --- Plain file resolution ---


def test_prompt_reads_plain_file(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("Remember: coffee first.\n")
    ext = MemoryExtension(path=tmp_path, project_discovery=False)
    result = ext.prompt({}, None)
    assert result is not None
    assert "# Memory" in result
    assert "Remember: coffee first." in result
    assert "Memory records may be stale." in result


def test_prompt_custom_header_and_footer(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("body")
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        header="## Custom Header",
        trust_footer="TRUST ME",
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert result.startswith("## Custom Header")
    assert result.endswith("TRUST ME")


# --- Project discovery ---


def test_prompt_uses_project_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_cwd = tmp_path / "workspace" / "my-proj"
    project_cwd.mkdir(parents=True)
    monkeypatch.chdir(project_cwd)

    key = sanitize_path(str(project_cwd))
    mem_dir = tmp_path / "mem"
    (mem_dir / key).mkdir(parents=True)
    (mem_dir / key / "MEMORY.md").write_text("project-specific")

    ext = MemoryExtension(path=mem_dir)
    result = ext.prompt({}, None)
    assert result is not None
    assert "project-specific" in result


def test_prompt_custom_project_key_fn(tmp_path: Path) -> None:
    mem_dir = tmp_path / "mem"
    (mem_dir / "proj-A").mkdir(parents=True)
    (mem_dir / "proj-A" / "MEMORY.md").write_text("body-A")

    ext = MemoryExtension(path=mem_dir, project_key_fn=lambda _p: "proj-A")
    result = ext.prompt({}, None)
    assert result is not None
    assert "body-A" in result


def test_prompt_project_key_fn_traversal_rejected(tmp_path: Path) -> None:
    ext = MemoryExtension(path=tmp_path, project_key_fn=lambda _p: "../etc")
    with pytest.raises(ValueError):
        ext.prompt({}, None)


def test_prompt_project_key_fn_backslash_rejected(tmp_path: Path) -> None:
    ext = MemoryExtension(path=tmp_path, project_key_fn=lambda _p: "a\\b")
    with pytest.raises(ValueError):
        ext.prompt({}, None)


def test_prompt_project_key_fn_empty_falls_back(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("fallback")
    ext = MemoryExtension(path=tmp_path, project_key_fn=lambda _p: "   ")
    result = ext.prompt({}, None)
    assert result is not None
    assert "fallback" in result


def test_prompt_project_key_fn_raises_falls_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "MEMORY.md").write_text("fallback body")

    def boom(_p: Path) -> str:
        raise RuntimeError("broken key fn")

    ext = MemoryExtension(path=tmp_path, project_key_fn=boom)
    with caplog.at_level(logging.DEBUG, logger="langchain_agentkit.extensions.memory.extension"):
        result = ext.prompt({}, None)
    assert result is not None
    assert "fallback body" in result


# --- Caps ---


def test_prompt_applies_max_lines_cap(tmp_path: Path) -> None:
    lines = "\n".join(f"line-{i}" for i in range(500))
    (tmp_path / "MEMORY.md").write_text(lines)
    ext = MemoryExtension(path=tmp_path, project_discovery=False, max_lines=10)
    result = ext.prompt({}, None)
    assert result is not None
    assert "line-9" in result
    assert "line-10" not in result
    assert "... [truncated]" in result


def test_prompt_applies_max_bytes_cap(tmp_path: Path) -> None:
    body = "x" * 5000
    (tmp_path / "MEMORY.md").write_text(body)
    ext = MemoryExtension(path=tmp_path, project_discovery=False, max_bytes=100, max_lines=10000)
    result = ext.prompt({}, None)
    assert result is not None
    assert "... [truncated]" in result
    # The body portion should be capped
    # Count x's in body portion — must be <= 100
    x_count = result.count("x")
    assert x_count <= 100


# --- Loader override ---


def test_prompt_loader_overrides_file(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("ignored")
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        loader=lambda: "loaded body",
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "loaded body" in result
    assert "ignored" not in result


def test_prompt_loader_returns_none_falls_through_to_none(tmp_path: Path) -> None:
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        loader=lambda: None,
    )
    # loader returning None means no body, no file either → None overall
    assert ext.prompt({}, None) is None


# --- extra_sources ---


def test_prompt_includes_extra_sources(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("main body")
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        extra_sources=[lambda: "extra1", lambda: "extra2"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "extra1" in result
    assert "extra2" in result


def test_prompt_extra_sources_skip_none_and_empty(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("main")
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        extra_sources=[lambda: None, lambda: "", lambda: "real"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "real" in result


def test_prompt_extra_sources_swallow_exceptions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "MEMORY.md").write_text("main")

    def boom() -> str:
        raise RuntimeError("bad source")

    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        extra_sources=[boom, lambda: "ok"],
    )
    with caplog.at_level(logging.DEBUG, logger="langchain_agentkit.extensions.memory.extension"):
        result = ext.prompt({}, None)
    assert result is not None
    assert "ok" in result


def test_prompt_only_extra_sources_no_file(tmp_path: Path) -> None:
    ext = MemoryExtension(
        path=tmp_path,
        project_discovery=False,
        extra_sources=[lambda: "only this"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "only this" in result


# --- Path expansion ---


def test_prompt_expands_user_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text("env body")
    monkeypatch.setenv("MEMDIR", str(mem_dir))
    ext = MemoryExtension(path="$MEMDIR", project_discovery=False)
    result = ext.prompt({}, None)
    assert result is not None
    assert "env body" in result
