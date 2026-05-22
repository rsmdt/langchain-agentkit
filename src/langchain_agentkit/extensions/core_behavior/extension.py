"""CoreBehaviorExtension — universal, domain-neutral agent guidance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolRuntime

_CORE_BEHAVIOR_BODY = (
    (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8").rstrip()
)


class CoreBehaviorExtension(Extension):
    """Contributes universal, domain-neutral agent guidance to the prompt.

    Tool-specific guidance (e.g. "prefer Read over cat", "Bash-only
    conventions") lives with the extension that contributes those tools
    — see :class:`FilesystemExtension`.
    """

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str:
        return _CORE_BEHAVIOR_BODY
