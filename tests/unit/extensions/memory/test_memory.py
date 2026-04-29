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


# --- Backend mode ---


class _FakeBackend:
    """Minimal BackendProtocol stand-in for memory-read tests."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.reads: list[str] = []

    async def read(self, path: str, offset: int = 0, limit: int = 2000):
        from langchain_agentkit.backends.results import ReadResult

        self.reads.append(path)
        if path not in self.files:
            return ReadResult(error="file_not_found", error_message=f"File not found: {path}")
        return ReadResult(content=self.files[path])

    async def read_bytes(self, path: str):  # pragma: no cover — unused here
        raise NotImplementedError

    async def write(self, path: str, content):  # pragma: no cover
        raise NotImplementedError

    async def edit(self, path, old_string, new_string, replace_all=False):  # pragma: no cover
        raise NotImplementedError

    async def glob(self, pattern, path="/"):  # pragma: no cover
        raise NotImplementedError

    async def grep(self, pattern, path=None, glob=None, ignore_case=False):  # pragma: no cover
        raise NotImplementedError

    async def execute(self, command, timeout=None, workdir=None):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.asyncio
async def test_setup_primes_cache_from_backend(tmp_path: Path) -> None:
    target = str(tmp_path / "MEMORY.md")
    backend = _FakeBackend({target: "backend body"})
    ext = MemoryExtension(path=tmp_path, project_discovery=False, backend=backend)

    # Before setup, no body cached.
    assert ext.prompt({}, None) is None

    await ext.setup()
    result = ext.prompt({}, None)
    assert result is not None
    assert "backend body" in result


@pytest.mark.asyncio
async def test_before_model_refreshes_cache(tmp_path: Path) -> None:
    target = str(tmp_path / "MEMORY.md")
    backend = _FakeBackend({target: "v1"})
    ext = MemoryExtension(path=tmp_path, project_discovery=False, backend=backend)
    await ext.setup()
    assert "v1" in (ext.prompt({}, None) or "")

    backend.files[target] = "v2"
    await ext.before_model(state={}, runtime=None)
    result = ext.prompt({}, None)
    assert result is not None
    assert "v2" in result
    assert "v1" not in result


@pytest.mark.asyncio
async def test_backend_falls_back_to_base_when_project_specific_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_cwd = tmp_path / "workspace" / "proj-x"
    project_cwd.mkdir(parents=True)
    monkeypatch.chdir(project_cwd)

    base = tmp_path / "mem"
    # Only the non-project base file exists in the backend.
    backend = _FakeBackend({str(base / "MEMORY.md"): "fallback content"})
    ext = MemoryExtension(path=base, backend=backend)
    await ext.setup()

    # Both candidates were attempted; project-specific first, then base.
    assert len(backend.reads) == 2
    result = ext.prompt({}, None)
    assert result is not None
    assert "fallback content" in result


@pytest.mark.asyncio
async def test_backend_returns_none_when_no_file_found(tmp_path: Path) -> None:
    backend = _FakeBackend({})
    ext = MemoryExtension(path=tmp_path, project_discovery=False, backend=backend)
    await ext.setup()
    assert ext.prompt({}, None) is None


@pytest.mark.asyncio
async def test_backend_applies_caps(tmp_path: Path) -> None:
    target = str(tmp_path / "MEMORY.md")
    body = "\n".join(f"line-{i}" for i in range(500))
    backend = _FakeBackend({target: body})
    ext = MemoryExtension(path=tmp_path, project_discovery=False, backend=backend, max_lines=10)
    await ext.setup()
    result = ext.prompt({}, None)
    assert result is not None
    assert "line-9" in result
    assert "line-10" not in result
    assert "... [truncated]" in result


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
