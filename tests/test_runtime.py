"""Tests for ToolRuntime."""

from langchain_agentkit.runtime import ToolRuntime


class TestToolRuntime:
    def test_config_property_returns_provided_config(self):
        config = {"configurable": {"thread_id": "abc"}}

        runtime = ToolRuntime(config=config)

        assert runtime.config is config

    def test_get_returns_extra_value(self):
        runtime = ToolRuntime(config={}, store="my_store")

        assert runtime.get("store") == "my_store"

    def test_get_returns_default_for_missing_key(self):
        runtime = ToolRuntime(config={})

        assert runtime.get("missing") is None
        assert runtime.get("missing", "fallback") == "fallback"

    def test_repr_without_extras(self):
        runtime = ToolRuntime(config={})

        assert "ToolRuntime" in repr(runtime)
        assert "extras" not in repr(runtime)

    def test_repr_with_extras(self):
        runtime = ToolRuntime(config={}, store="s")

        result = repr(runtime)

        assert "ToolRuntime" in result
        assert "extras" in result
        assert "store" in result
