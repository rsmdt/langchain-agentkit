"""Integration tests for ``AgentKit(preset="full")`` batteries-included kit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from langchain_agentkit import (
    AgentKit,
    TasksExtension,
)

if TYPE_CHECKING:
    from pathlib import Path
from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension
from langchain_agentkit.extensions.memory import MemoryExtension


class _StubExtension:
    """Minimal extension to verify caller-supplied extensions are appended."""

    def __init__(self) -> None:
        self._tools: list[Any] = []

    @property
    def tools(self) -> list[Any]:
        return self._tools

    def prompt(self, state: dict, runtime: Any = None) -> str | None:
        return None


class TestFullPresetDefaults:
    def test_preset_full_seeds_core_behavior_and_tasks(self) -> None:
        kit = AgentKit(preset="full")
        types = [type(e) for e in kit.extensions]
        assert CoreBehaviorExtension in types
        assert TasksExtension in types
        assert MemoryExtension not in types

    def test_preset_full_order_is_core_then_tasks(self) -> None:
        kit = AgentKit(preset="full")
        types = [type(e) for e in kit.extensions]
        seeded = [t for t in types if t in {CoreBehaviorExtension, TasksExtension}]
        assert seeded == [CoreBehaviorExtension, TasksExtension]

    def test_no_preset_leaves_extensions_empty(self) -> None:
        kit = AgentKit()
        assert kit.extensions == []

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="preset"):
            AgentKit(preset="bogus")

    def test_compose_has_non_empty_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(preset="full")
        composition = kit.compose({})
        assert composition.prompt.strip()

    def test_base_prompt_forwarded(self) -> None:
        kit = AgentKit(preset="full", prompt="You are a code reviewer.")
        assert kit.base_prompt == "You are a code reviewer."


class TestPresetAndUserExtensions:
    def test_user_extensions_appended_after_seeded(self) -> None:
        extra = _StubExtension()
        kit = AgentKit(preset="full", extensions=[extra])
        assert extra in kit.extensions
        # Prepended seeds must come first.
        assert type(kit.extensions[0]) is CoreBehaviorExtension
        assert kit.extensions[-1] is extra

    def test_name_forwarded(self) -> None:
        kit = AgentKit(preset="full", name="researcher")
        assert kit._name == "researcher"
