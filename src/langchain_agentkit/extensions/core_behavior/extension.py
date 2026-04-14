"""CoreBehaviorExtension — universal, domain-neutral agent guidance plus <env>."""

from __future__ import annotations

import os
import platform as _platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_CORE_BEHAVIOR_BODY = (_PROMPTS_DIR / "core_behavior.md").read_text(encoding="utf-8").rstrip()
_ENV_TEMPLATE = (_PROMPTS_DIR / "env.md").read_text(encoding="utf-8").rstrip()

_WORKTREE_LINE = "This is a git worktree — do NOT cd to original repo"


def _find_git_marker(start: Path) -> Path | None:
    """Return the `.git` path (file or dir) for the nearest ancestor, or None."""
    try:
        current = start.resolve()
    except OSError:
        current = start
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.exists():
            return marker
    return None


class CoreBehaviorExtension(Extension):
    """Contributes universal agent guidance plus an auto-detected <env> block.

    Args:
        cwd: Optional callable returning the current working directory. Invoked
            on every ``prompt()`` call so per-turn changes are reflected.
        include_env: When False, the ``<env>`` block is omitted and ``prompt()``
            returns only the universal-guidance body.
    """

    prompt_cache_scope = "static"

    def __init__(
        self,
        cwd: Callable[[], Path] | None = None,
        include_env: bool = True,
    ) -> None:
        self._cwd = cwd
        self._include_env = include_env

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        if not self._include_env:
            return _CORE_BEHAVIOR_BODY
        return f"{_CORE_BEHAVIOR_BODY}\n\n{self._render_env()}"

    # --- internals ---

    def _render_env(self) -> str:
        cwd = self._cwd() if self._cwd is not None else Path.cwd()
        marker = _find_git_marker(cwd)
        is_git = marker is not None
        worktree_line = _WORKTREE_LINE if marker is not None and marker.is_file() else ""
        shell = Path(os.environ.get("SHELL", "")).name or ""
        return _ENV_TEMPLATE.format(
            cwd=cwd,
            is_git=is_git,
            platform=f"{_platform.system()} {_platform.release()}",
            shell=shell,
            os_version=_platform.platform(),
            worktree_line=worktree_line,
        )
