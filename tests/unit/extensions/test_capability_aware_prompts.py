"""Tests for capability-aware prompt composition (I3)."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension


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


def test_core_behavior_bash_with_specialized_tools() -> None:
    kit = _kit(["Bash", "Read", "Grep"], CoreBehaviorExtension())
    composition = kit.compose({}, None)
    assert "Dedicated file tools are available" in composition.prompt
    assert "Reserve `Bash`" in composition.prompt


def test_core_behavior_bash_only() -> None:
    kit = _kit(["Bash"], CoreBehaviorExtension())
    composition = kit.compose({}, None)
    assert "Shell-only environment" in composition.prompt
    assert "Dedicated file tools are available" not in composition.prompt


def test_core_behavior_no_bash_no_appendix() -> None:
    kit = _kit(["Read", "Grep"], CoreBehaviorExtension())
    composition = kit.compose({}, None)
    assert "Shell-only environment" not in composition.prompt
    assert "Dedicated file tools are available" not in composition.prompt


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
