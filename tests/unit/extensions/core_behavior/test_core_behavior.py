"""Tests for CoreBehaviorExtension."""

from __future__ import annotations

from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension


class TestCoreBehaviorExtensionConstruction:
    def test_defaults(self):
        ext = CoreBehaviorExtension()
        assert ext.tools == []
        assert ext.state_schema is None


class TestPromptBody:
    def test_prompt_returns_string(self):
        ext = CoreBehaviorExtension()
        out = ext.prompt({}, None)
        assert isinstance(out, str)
        assert out.strip() != ""


class TestDependencies:
    def test_no_dependencies(self):
        ext = CoreBehaviorExtension()
        assert ext.dependencies() == []
