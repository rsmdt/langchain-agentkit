"""Tests for HITLMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.middleware.hitl import HITLMiddleware

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestInit:
    def test_bool_true_expands_to_all_decisions(self):
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        assert "send_email" in mw.interrupt_on
        assert mw.interrupt_on["send_email"].allowed_decisions == [
            "approve",
            "edit",
            "reject",
        ]

    def test_bool_false_excluded(self):
        mw = HITLMiddleware(interrupt_on={"search": False, "send_email": True})

        assert "search" not in mw.interrupt_on
        assert "send_email" in mw.interrupt_on

    def test_explicit_config_preserved(self):
        config = {"allowed_decisions": ["approve", "reject"]}
        mw = HITLMiddleware(interrupt_on={"send_email": config})

        assert mw.interrupt_on["send_email"].allowed_decisions == ["approve", "reject"]

    def test_empty_interrupt_on(self):
        mw = HITLMiddleware(interrupt_on={})

        assert mw.interrupt_on == {}


class TestMiddlewareProtocol:
    def test_tools_returns_empty_list(self):
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        assert mw.tools == []

    def test_prompt_returns_none(self):
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        result = mw.prompt({}, _TEST_RUNTIME)

        assert result is None


class TestWrapToolCall:
    def test_auto_approved_tool_executes_normally(self):
        """Tools not in interrupt_on pass through to execute."""
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        mock_request = MagicMock()
        mock_request.tool_call = {"name": "search", "args": {"q": "test"}, "id": "call_1"}
        expected = ToolMessage(content="result", tool_call_id="call_1")
        mock_execute = MagicMock(return_value=expected)

        result = mw.wrap_tool_call(mock_request, mock_execute)

        mock_execute.assert_called_once_with(mock_request)
        assert result == expected

    def test_has_wrap_tool_call_method(self):
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        assert callable(getattr(mw, "wrap_tool_call", None))


class TestAgentIntegration:
    def test_agent_detects_wrap_tool_call_from_middleware(self):
        """Agent metaclass should detect wrap_tool_call on middleware."""
        mw = HITLMiddleware(interrupt_on={"send_email": True})

        assert hasattr(mw, "wrap_tool_call")
        assert callable(mw.wrap_tool_call)
