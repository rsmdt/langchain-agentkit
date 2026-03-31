"""Tests for ToolRuntime integration with langgraph.prebuilt."""

from langgraph.prebuilt import ToolRuntime


class TestToolRuntime:
    def test_config_field_returns_provided_config(self):
        config = {"configurable": {"thread_id": "abc"}}

        runtime = ToolRuntime(
            state={},
            context=None,
            config=config,
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=None,
        )

        assert runtime.config is config

    def test_state_field_returns_provided_state(self):
        state = {"messages": []}

        runtime = ToolRuntime(
            state=state,
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=None,
        )

        assert runtime.state is state

    def test_store_field(self):
        runtime = ToolRuntime(
            state={},
            context=None,
            config={},
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=None,
        )

        assert runtime.store is None

    def test_available_from_langgraph(self):
        """ToolRuntime is a langgraph type — import directly from langgraph.prebuilt."""
        from langgraph.prebuilt import ToolRuntime as LangGraphToolRuntime

        assert LangGraphToolRuntime is ToolRuntime
