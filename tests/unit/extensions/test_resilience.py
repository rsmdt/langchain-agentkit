"""Tests for ResilienceExtension (Layer 1: write-time prevention)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from langchain_agentkit.extensions.resilience import (
    ResilienceExtension,
    ToolErrorEvent,
)


def _make_request(tool_name: str = "delegate", tool_call_id: str = "call_1") -> MagicMock:
    """Build a ToolCallRequest-shaped stub with the right tool_call dict."""
    request = MagicMock()
    request.tool_call = {
        "name": tool_name,
        "args": {"message": "hi"},
        "id": tool_call_id,
    }
    return request


class TestHookRegistration:
    def test_wrap_tool_discovered_as_named_hook(self):
        assert ("wrap", "tool") in ResilienceExtension._get_named_hooks()


class TestPassthrough:
    async def test_successful_tool_call_returns_handler_result(self):
        ext = ResilienceExtension()
        request = _make_request()
        expected = ToolMessage(content="ok", tool_call_id="call_1")

        result = await ext.wrap_tool(
            state=request, handler=AsyncMock(return_value=expected), runtime=None
        )

        assert result is expected


class TestErrorCatch:
    async def test_unhandled_exception_becomes_tool_message(self):
        ext = ResilienceExtension()
        request = _make_request(tool_name="delegate", tool_call_id="call_xyz")
        handler = AsyncMock(side_effect=RuntimeError("boom"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_xyz"
        assert result.name == "delegate"
        assert result.status == "error"
        assert "RuntimeError" in result.content
        assert "boom" in result.content
        agentkit_meta = result.additional_kwargs.get("agentkit", {})
        assert agentkit_meta["synthesized"] is True
        assert agentkit_meta["reason"] == "tool_error"
        assert agentkit_meta["exc_type"] == "RuntimeError"

    async def test_type_error_pattern_matches_original_bug(self):
        """Regression: the parent_llm_getter TypeError path that created
        orphan AIMessages in the first place must be caught here."""
        ext = ResilienceExtension()
        request = _make_request(tool_name="Agent", tool_call_id="call_lIRriSLL")
        handler = AsyncMock(side_effect=TypeError("'NoneType' object is not callable"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_lIRriSLL"

    async def test_custom_template_controls_message(self):
        ext = ResilienceExtension(
            tool_error_template=lambda exc, name: f"<<{name}:{type(exc).__name__}>>",
            include_exception_message=False,
        )
        request = _make_request(tool_name="delegate")
        handler = AsyncMock(side_effect=ValueError("secret"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert result.content == "<<delegate:ValueError>>"
        assert "secret" not in result.content

    async def test_empty_exception_message_omitted(self):
        ext = ResilienceExtension()
        request = _make_request()
        handler = AsyncMock(side_effect=RuntimeError(""))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert result.content == "[tool 'delegate' failed: RuntimeError]"


class TestReraisePolicy:
    async def test_tool_exception_is_reraised(self):
        """ToolException is part of the typed error contract — let ToolNode handle it."""
        ext = ResilienceExtension()
        request = _make_request()
        handler = AsyncMock(side_effect=ToolException("user-visible"))

        with pytest.raises(ToolException, match="user-visible"):
            await ext.wrap_tool(state=request, handler=handler, runtime=None)

    async def test_cancelled_error_is_reraised(self):
        """CancelledError is a control signal — never swallow it."""
        ext = ResilienceExtension()
        request = _make_request()
        handler = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await ext.wrap_tool(state=request, handler=handler, runtime=None)

    async def test_base_exception_escapes(self):
        """KeyboardInterrupt / SystemExit must not be caught."""
        ext = ResilienceExtension()
        request = _make_request()
        handler = AsyncMock(side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            await ext.wrap_tool(state=request, handler=handler, runtime=None)


class TestTelemetry:
    async def test_callback_fires_with_event(self):
        events: list[ToolErrorEvent] = []
        ext = ResilienceExtension(on_tool_error_caught=events.append)
        request = _make_request(tool_name="Agent", tool_call_id="call_abc")
        handler = AsyncMock(side_effect=RuntimeError("db down"))

        await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert len(events) == 1
        event = events[0]
        assert event.tool_call_id == "call_abc"
        assert event.tool_name == "Agent"
        assert event.exc_type == "RuntimeError"
        assert event.exc_message == "db down"

    async def test_callback_failure_does_not_break_execution(self):
        """A broken telemetry sink must never corrupt the tool pipeline."""

        def broken_callback(event: ToolErrorEvent) -> None:
            raise RuntimeError("sink exploded")

        ext = ResilienceExtension(on_tool_error_caught=broken_callback)
        request = _make_request()
        handler = AsyncMock(side_effect=ValueError("inner"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    async def test_warn_log_emitted(self, caplog):
        ext = ResilienceExtension()
        request = _make_request(tool_name="Agent", tool_call_id="call_log")
        handler = AsyncMock(side_effect=RuntimeError("logged"))

        with caplog.at_level(logging.WARNING, logger="langchain_agentkit.extensions.resilience"):
            await ext.wrap_tool(state=request, handler=handler, runtime=None)

        matches = [r for r in caplog.records if "call_log" in r.getMessage()]
        assert matches, "expected a WARN record referencing tool_call_id"


class TestDefensiveExtraction:
    async def test_missing_tool_call_does_not_crash(self):
        """Unexpected state shape must not mask the original failure."""
        ext = ResilienceExtension()
        request = MagicMock(spec=[])  # no tool_call attr
        handler = AsyncMock(side_effect=RuntimeError("boom"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == ""
        assert result.name == ""

    async def test_object_shaped_tool_call(self):
        """wrap_tool sometimes sees an object-shaped tool_call, not a dict."""
        ext = ResilienceExtension()

        class ToolCall:
            id = "call_obj"
            name = "Agent"
            args = {}

        request = MagicMock()
        request.tool_call = ToolCall()
        handler = AsyncMock(side_effect=RuntimeError("x"))

        result = await ext.wrap_tool(state=request, handler=handler, runtime=None)

        assert result.tool_call_id == "call_obj"
        assert result.name == "Agent"
