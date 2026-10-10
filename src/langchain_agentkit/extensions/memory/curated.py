"""CuratedMemory — an operator-authored document, always in context.

Reads a memory file from a configurable location and contributes its
contents to the durable system prompt.  Discovery mirrors a per-project
layout by default: ``<path>/<sanitized-cwd>/<filename>``.  Custom layouts
are supported via ``project_key_fn``.

Two modes:

1. **Local filesystem** (``backend=None``) — reads synchronously inside
   ``body()`` via ``pathlib.Path``. No async setup needed.
2. **Backend-driven** (``backend=<FilesystemProtocol>``) — reads
   asynchronously. An initial load runs in ``setup()``; ``refresh()``
   updates a cached body that ``body()`` returns synchronously.

Missing files are not errors; they simply contribute nothing.  Reads that
fail for any *other* reason are logged as warnings, because a memory
document that silently vanishes is indistinguishable from one that was
never configured.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.extensions.memory.strategy import MemoryStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_agentkit.backends.protocol import FilesystemProtocol

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")
_TRUNCATED_MARKER = "... [truncated]"


# Port of the reference harness ``sanitizePath()``.
def sanitize_path(name: str, *, max_length: int = 200) -> str:
    replaced = _SANITIZE_RE.sub("-", name)
    if len(replaced) <= max_length:
        return replaced
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return replaced[:max_length] + "-" + digest


def _default_project_key_fn(cwd: Path) -> str:
    return sanitize_path(str(cwd))


@dataclass(kw_only=True)
class CuratedMemory(MemoryStrategy):
    """A markdown document injected whole into the system prompt.

    All parameters are keyword-only and defaulted.  See the module
    docstring for resolution order.

    Args:
        backend: Optional :class:`FilesystemProtocol` for remote filesystem
            reads. When provided, ``setup()`` performs the initial async
            load and ``refresh()`` reloads every model call. When ``None``,
            ``body()`` reads the local filesystem synchronously.
    """

    path: str | Path = "~/.agents/memory"
    filename: str = "MEMORY.md"
    project_discovery: bool = True
    project_key_fn: Callable[[Path], str] | None = None
    project_key_max_length: int = 200
    max_lines: int = 200
    max_bytes: int = 25_000
    backend: FilesystemProtocol | None = None

    def __post_init__(self) -> None:
        self._cached_body: str | None = None

    # --- MemoryStrategy protocol ---

    @override
    async def setup(self) -> None:
        if self.backend is not None:
            self._cached_body = await self._load_body_async()

    @override
    async def refresh(self, *, state: dict[str, Any], runtime: Any) -> None:
        if self.backend is not None:
            self._cached_body = await self._load_body_async()

    @override
    def body(self) -> str | None:
        if self.backend is not None:
            return self._cached_body
        return self._load_body_sync()

    # --- Internals ---

    def _resolve_base_path(self) -> Path:
        raw = str(self.path)
        expanded = os.path.expandvars(os.path.expanduser(raw))
        return Path(expanded)

    def _resolve_project_key(self) -> str | None:
        if not self.project_discovery:
            return None
        fn = self.project_key_fn or _default_project_key_fn
        try:
            key = fn(Path.cwd())
        except Exception:
            logger.debug("CuratedMemory: project_key_fn raised; falling back", exc_info=True)
            return None
        if not isinstance(key, str):
            logger.debug("CuratedMemory: project_key_fn returned non-string; falling back")
            return None
        # Directory-traversal guard: reject separators.  This is a real
        # error, not a silent fallback — a key fn producing path
        # separators is a bug or an attempted escape.
        if "/" in key or "\\" in key:
            raise ValueError(f"project_key_fn returned a key containing path separators: {key!r}")
        stripped = key.strip()
        if not stripped:
            return None
        return stripped

    def _candidate_paths(self) -> list[Path]:
        base = self._resolve_base_path()
        key = self._resolve_project_key()
        candidates: list[Path] = []
        if key is not None:
            candidates.append(base / key / self.filename)
        candidates.append(base / self.filename)
        return candidates

    def _load_body_sync(self) -> str | None:
        for candidate in self._candidate_paths():
            if not candidate.is_file():
                continue
            try:
                raw = candidate.read_text(encoding="utf-8")
            except OSError:
                logger.warning(
                    "CuratedMemory: could not read %s; memory contributes nothing",
                    candidate,
                    exc_info=True,
                )
                continue
            if raw:
                return self._apply_caps(raw)
        return None

    async def _load_body_async(self) -> str | None:
        assert self.backend is not None  # guarded by callers
        for candidate in self._candidate_paths():
            result = await self.backend.read(str(candidate), limit=self.max_lines + 1)
            if result.error is not None:
                # A missing file is the ordinary "no memory yet" case. Any
                # other error means a configured document silently
                # disappeared — most often a backend rooted somewhere that
                # does not contain ``path`` (an OSBackend scoped to cwd
                # rejects an outside path as ``permission_denied``).
                if result.error != "file_not_found":
                    logger.warning(
                        "CuratedMemory: backend read failed for %s (%s); memory "
                        "contributes nothing. Check that the backend root contains "
                        "the memory path.",
                        candidate,
                        result.error,
                    )
                continue
            if result.content:
                return self._apply_caps(result.content)
        return None

    def _apply_caps(self, text: str) -> str:
        truncated = False

        lines = text.splitlines()
        if len(lines) > self.max_lines:
            lines = lines[: self.max_lines]
            truncated = True
        capped = "\n".join(lines)

        encoded = capped.encode("utf-8")
        if len(encoded) > self.max_bytes:
            encoded = encoded[: self.max_bytes]
            # Decode defensively — may drop a trailing partial multibyte char.
            capped = encoded.decode("utf-8", errors="ignore")
            truncated = True

        if truncated:
            capped = capped.rstrip("\n") + "\n" + _TRUNCATED_MARKER
        return capped
