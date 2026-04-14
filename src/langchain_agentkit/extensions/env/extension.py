"""EnvExtension — auto-detected ``<env>`` block contributed to the prompt."""

from __future__ import annotations

import os
import platform as _platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt import ToolRuntime

_ENV_TEMPLATE = (Path(__file__).parent / "prompts" / "env.md").read_text(encoding="utf-8").rstrip()

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


class EnvExtension(Extension):
    """Contributes an auto-detected ``<env>`` block describing the runtime.

    The block reports cwd, git detection, platform, shell, and OS version,
    and is rendered on every ``prompt()`` call so per-turn changes are
    reflected.

    Args:
        cwd: Optional callable returning the current working directory. Invoked
            on every ``prompt()`` call. Defaults to :func:`pathlib.Path.cwd`.
    """

    def __init__(self, cwd: Callable[[], Path] | None = None) -> None:
        self._cwd = cwd

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
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
