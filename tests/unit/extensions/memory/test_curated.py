"""Tests for the CuratedMemory strategy."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.extensions.memory import CuratedMemory, sanitize_path

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


# --- No I/O in init ---


def test_init_no_filesystem_io(tmp_path: Path) -> None:
    # Even with a non-existent path, construction must succeed.
    strategy = CuratedMemory(path=tmp_path / "nonexistent", project_discovery=False)
    assert strategy.tools == []


def test_body_returns_none_when_no_file(tmp_path: Path) -> None:
    assert CuratedMemory(path=tmp_path, project_discovery=False).body() is None


def test_reminder_is_none(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("body")
    # A curated document is byte-stable, so it belongs in the durable
    # region only — it never contributes to the reminder channel.
    assert CuratedMemory(path=tmp_path, project_discovery=False).reminder() is None


# --- Plain file resolution ---


def test_body_reads_plain_file(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("Remember: coffee first.\n")
    strategy = CuratedMemory(path=tmp_path, project_discovery=False)
    assert strategy.body() == "Remember: coffee first."


# --- Project discovery ---


def test_body_uses_project_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_cwd = tmp_path / "workspace" / "my-proj"
    project_cwd.mkdir(parents=True)
    monkeypatch.chdir(project_cwd)

    key = sanitize_path(str(project_cwd))
    mem_dir = tmp_path / "mem"
    (mem_dir / key).mkdir(parents=True)
    (mem_dir / key / "MEMORY.md").write_text("project-specific")

    assert CuratedMemory(path=mem_dir).body() == "project-specific"


def test_body_custom_project_key_fn(tmp_path: Path) -> None:
    mem_dir = tmp_path / "mem"
    (mem_dir / "proj-A").mkdir(parents=True)
    (mem_dir / "proj-A" / "MEMORY.md").write_text("body-A")

    strategy = CuratedMemory(path=mem_dir, project_key_fn=lambda _p: "proj-A")
    assert strategy.body() == "body-A"


def test_project_key_fn_traversal_rejected(tmp_path: Path) -> None:
    strategy = CuratedMemory(path=tmp_path, project_key_fn=lambda _p: "../etc")
    with pytest.raises(ValueError):
        strategy.body()


def test_project_key_fn_backslash_rejected(tmp_path: Path) -> None:
    strategy = CuratedMemory(path=tmp_path, project_key_fn=lambda _p: "a\\b")
    with pytest.raises(ValueError):
        strategy.body()


def test_project_key_fn_empty_falls_back(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("fallback")
    strategy = CuratedMemory(path=tmp_path, project_key_fn=lambda _p: "   ")
    assert strategy.body() == "fallback"


def test_project_key_fn_raises_falls_back(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("fallback body")

    def boom(_p: Path) -> str:
        raise RuntimeError("broken key fn")

    assert CuratedMemory(path=tmp_path, project_key_fn=boom).body() == "fallback body"


# --- Caps ---


def test_applies_max_lines_cap(tmp_path: Path) -> None:
    lines = "\n".join(f"line-{i}" for i in range(500))
    (tmp_path / "MEMORY.md").write_text(lines)
    result = CuratedMemory(path=tmp_path, project_discovery=False, max_lines=10).body()
    assert result is not None
    assert "line-9" in result
    assert "line-10" not in result
    assert "... [truncated]" in result


def test_applies_max_bytes_cap(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("x" * 5000)
    result = CuratedMemory(
        path=tmp_path, project_discovery=False, max_bytes=100, max_lines=10000
    ).body()
    assert result is not None
    assert "... [truncated]" in result
    assert result.count("x") <= 100


# --- Backend mode ---


class _FakeBackend:
    """Minimal FilesystemProtocol stand-in for memory-read tests."""

    def __init__(self, files: dict[str, str], *, error: str | None = None) -> None:
        self.files = files
        self.error = error
        self.reads: list[str] = []

    async def read(self, path: str, offset: int = 0, limit: int = 2000):
        from langchain_agentkit.backends.results import ReadResult

        self.reads.append(path)
        if self.error is not None:
            return ReadResult(error=self.error, error_message=f"{self.error}: {path}")
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
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=backend)

    # Before setup, no body cached.
    assert strategy.body() is None

    await strategy.setup()
    assert strategy.body() == "backend body"


@pytest.mark.asyncio
async def test_refresh_reloads_cache(tmp_path: Path) -> None:
    target = str(tmp_path / "MEMORY.md")
    backend = _FakeBackend({target: "v1"})
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=backend)
    await strategy.setup()
    assert strategy.body() == "v1"

    backend.files[target] = "v2"
    await strategy.refresh(state={}, runtime=None)
    assert strategy.body() == "v2"


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
    strategy = CuratedMemory(path=base, backend=backend)
    await strategy.setup()

    # Both candidates were attempted; project-specific first, then base.
    assert len(backend.reads) == 2
    assert strategy.body() == "fallback content"


@pytest.mark.asyncio
async def test_backend_returns_none_when_no_file_found(tmp_path: Path) -> None:
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=_FakeBackend({}))
    await strategy.setup()
    assert strategy.body() is None


@pytest.mark.asyncio
async def test_backend_missing_file_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=_FakeBackend({}))
    with caplog.at_level(logging.WARNING):
        await strategy.setup()
    assert caplog.records == []


@pytest.mark.asyncio
async def test_backend_permission_denied_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A backend rooted outside the memory path rejects the read. That must
    # be loud: the document silently contributing nothing is otherwise
    # indistinguishable from never having configured one.
    backend = _FakeBackend({}, error="permission_denied")
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=backend)
    with caplog.at_level(logging.WARNING):
        await strategy.setup()
    assert strategy.body() is None
    assert any("permission_denied" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_backend_applies_caps(tmp_path: Path) -> None:
    target = str(tmp_path / "MEMORY.md")
    body = "\n".join(f"line-{i}" for i in range(500))
    backend = _FakeBackend({target: body})
    strategy = CuratedMemory(path=tmp_path, project_discovery=False, backend=backend, max_lines=10)
    await strategy.setup()
    result = strategy.body()
    assert result is not None
    assert "line-9" in result
    assert "line-10" not in result
    assert "... [truncated]" in result


# --- Path expansion ---


def test_expands_user_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text("env body")
    monkeypatch.setenv("MEMDIR", str(mem_dir))
    assert CuratedMemory(path="$MEMDIR", project_discovery=False).body() == "env body"
