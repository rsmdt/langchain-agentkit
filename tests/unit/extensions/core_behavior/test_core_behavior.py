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

    def test_body_is_domain_neutral(self):
        ext = CoreBehaviorExtension()
        body = ext.prompt({}, None).lower()
        for forbidden in ("code", "test", "commit", "pull request", " pr ", "software"):
            assert forbidden not in body, f"found forbidden token: {forbidden!r}"

    def test_body_under_2kb(self):
        ext = CoreBehaviorExtension()
        body = ext.prompt({}, None)
        assert len(body.encode("utf-8")) <= 2048

    def test_no_env_block(self):
        ext = CoreBehaviorExtension()
        out = ext.prompt({}, None)
        assert "<env>" not in out


class TestDependencies:
    def test_no_dependencies(self):
        ext = CoreBehaviorExtension()
        assert ext.dependencies() == []
