"""Tests for capability-aware prompt composition.

Verifies that the kit's :class:`PromptComposition` machinery passes the
composed tool-name set to each extension's ``prompt()`` so they can
emit guidance keyed on which tools are actually registered.

A smoke test against :class:`FilesystemExtension` confirms the wiring
end-to-end (its tool-preference appendix only appears when ``Bash`` is
in the composed set). Detailed appendix-branching tests live in
``tests/unit/extensions/test_filesystem.py`` since they're properties
of :class:`FilesystemExtension`, not the kit infrastructure.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.filesystem import FilesystemExtension


class _FakeTool(BaseTool):
    """Minimal BaseTool whose name can be set at construction."""

    name: str = Field(default="unnamed")
    description: str = "fake"

    def _run(self, *_args: Any, **_kwargs: Any) -> str:
        return ""


def _kit(tool_names: list[str], extension: Extension) -> AgentKit:
    return AgentKit(
        extensions=[extension],
        tools=[_FakeTool(name=n, description=f"fake {n}") for n in tool_names],
    )


def test_kit_forwards_tool_names_to_filesystem_extension() -> None:
    """Kit composition reaches FilesystemExtension and triggers its appendix."""
    kit = _kit(["Bash"], FilesystemExtension())
    composition = kit.compose({}, None)
    # Bash + extension-contributed specialized tools → one-line preference directive.
    assert "Prefer the dedicated file tools over Bash" in composition.prompt


def test_legacy_prompt_signature_still_works() -> None:
    """Extensions written without the ``tools`` kwarg must keep working."""

    class LegacyExt(Extension):
        def prompt(self, state: dict[str, Any], runtime: Any = None) -> str:
            return "legacy guidance"

    kit = _kit(["Bash"], LegacyExt())
    composition = kit.compose({}, None)
    assert "legacy guidance" in composition.prompt


def test_extensions_receive_composed_tool_names() -> None:
    seen: dict[str, frozenset[str]] = {}

    class SpyExt(Extension):
        def prompt(
            self,
            state: dict[str, Any],
            runtime: Any = None,
            *,
            tools: frozenset[str] = frozenset(),
        ) -> str | None:
            seen["tools"] = tools
            return None

    kit = _kit(["Bash", "Grep"], SpyExt())
    kit.compose({}, None)
    assert seen["tools"] == frozenset({"Bash", "Grep"})
