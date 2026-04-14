"""CoreBehaviorExtension — universal, domain-neutral agent guidance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolRuntime

_CORE_BEHAVIOR_BODY = (
    (Path(__file__).parent / "prompts" / "core_behavior.md").read_text(encoding="utf-8").rstrip()
)


class CoreBehaviorExtension(Extension):
    """Contributes universal, domain-neutral agent guidance to the prompt."""

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        return _CORE_BEHAVIOR_BODY
