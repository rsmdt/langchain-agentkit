"""MemoryExtension — surface persistent memory content to the agent each turn.

Reads a memory file from a configurable location and contributes its
contents to the system prompt.  Discovery mirrors a per-project layout
by default: ``<path>/<sanitized-cwd>/<filename>``.  Custom layouts are
supported via ``project_key_fn``.

Two modes:

1. **Local filesystem** (``backend=None``) — reads synchronously inside
   ``prompt()`` via ``pathlib.Path``. No async setup needed.
2. **Backend-driven** (``backend=<BackendProtocol>``) — reads
   asynchronously. An initial load runs in ``setup()``; each turn's
   ``before_model`` hook refreshes a cached body that ``prompt()``
   returns synchronously.

Missing files are not errors; they simply contribute nothing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_agentkit.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9]")
_TRUNCATED_MARKER = "... [truncated]"


def sanitize_path(name: str, *, max_length: int = 200) -> str:
    """Port of the reference harness ``sanitizePath()``.

    Replace non-alphanumerics with ``-``.  If the result fits within
    ``max_length``, return it unchanged; otherwise truncate and append a
    12-character SHA-256 suffix derived from the original ``name`` to
    keep the key unique.
    """
    replaced = _SANITIZE_RE.sub("-", name)
    if len(replaced) <= max_length:
        return replaced
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return replaced[:max_length] + "-" + digest


def _default_project_key_fn(cwd: Path) -> str:
    return sanitize_path(str(cwd))


@dataclass(kw_only=True)
class MemoryExtension(Extension):
    """Contribute persistent memory content to the agent's system prompt.

    All parameters are keyword-only and defaulted.  See module docstring
    for resolution order.

    Args:
        backend: Optional :class:`BackendProtocol` for remote filesystem
            reads. When provided, ``setup()`` performs the initial async
            load and ``before_model`` refreshes the cached body every
            turn. When ``None``, ``prompt()`` reads the local filesystem
            synchronously each turn.
    """

    path: str | Path = "~/.agents/memory"
    filename: str = "MEMORY.md"
    project_discovery: bool = True
    project_key_fn: Callable[[Path], str] | None = None
    project_key_max_length: int = 200
    max_lines: int = 200
    max_bytes: int = 25_000
    header: str = "# Memory"
    extra_sources: list[Callable[[], str | None]] = field(default_factory=list)
    trust_footer: str = (
        "Memory records may be stale. Verify facts against current state before acting on them."
    )
    backend: BackendProtocol | None = None

    def __post_init__(self) -> None:
        self._cached_body: str | None = None

    # --- Extension protocol ---

    @property
    def tools(self) -> list[Any]:
        return []

    @property
    def state_schema(self) -> type | None:
        return None

    async def setup(self, **_: Any) -> None:  # type: ignore[override]
        """Prime the backend cache so the first turn sees a body."""
        if self.backend is not None:
            self._cached_body = await self._load_body_async()

    async def before_model(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        """Refresh the cached memory body per turn when a backend is set."""
        if self.backend is not None:
            self._cached_body = await self._load_body_async()
        return None

    def prompt(self, state: dict[str, Any], runtime: Any | None = None) -> str | None:
        body = self._get_body()
        extras = self._collect_extras()

        if body is None and not extras:
            return None

        sections: list[str] = [self.header, ""]
        if body is not None:
            sections.append(body)
        if extras:
            if body is not None:
                sections.append("")
            sections.append("\n\n".join(extras))
        sections.append("")
        sections.append(self.trust_footer)
        return "\n".join(sections)

    # --- Internals ---

    def _get_body(self) -> str | None:
        if self.backend is not None:
            return self._cached_body
        return self._load_body_sync()

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
            logger.debug("MemoryExtension: project_key_fn raised; falling back", exc_info=True)
            return None
        if not isinstance(key, str):
            logger.debug("MemoryExtension: project_key_fn returned non-string; falling back")
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
                logger.debug("MemoryExtension: failed to read %s", candidate, exc_info=True)
                continue
            if raw:
                return self._apply_caps(raw)
        return None

    async def _load_body_async(self) -> str | None:
        assert self.backend is not None  # guarded by callers
        for candidate in self._candidate_paths():
            try:
                raw = await self.backend.read(str(candidate), limit=self.max_lines + 1)
            except Exception:
                logger.debug(
                    "MemoryExtension: backend read failed for %s", candidate, exc_info=True
                )
                continue
            if raw:
                return self._apply_caps(raw)
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

    def _collect_extras(self) -> list[str]:
        out: list[str] = []
        for source in self.extra_sources:
            try:
                value = source()
            except Exception:
                logger.debug(
                    "MemoryExtension: extra_source raised; skipping",
                    exc_info=True,
                )
                continue
            if value is None or value == "":
                continue
            out.append(value)
        return out
