"""Tests for ResilienceExtension (Layer 1: write-time prevention)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from langchain_agentkit.extensions.resilience import (
    OrphanRepairEvent,
    ResilienceExtension,
    ToolErrorEvent,
)


def _ai_with_calls(*calls: tuple[str, str], content: str = "") -> AIMessage:
    """AIMessage with structured tool_calls entries. ``calls`` is (id, name) pairs."""
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": cid, "name": name, "args": {}, "type": "tool_call"} for cid, name in calls
        ],
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


class TestOrphanRepair:
    """Layer 2: read-time repair via wrap_model."""

    async def test_passthrough_when_no_orphans(self):
        ext = ResilienceExtension()
        messages = [
            HumanMessage(content="hi"),
            _ai_with_calls(("call_1", "search")),
            ToolMessage(content="result", tool_call_id="call_1", name="search"),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": [AIMessage(content="done")]}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        assert captured[0] == messages

    async def test_repairs_single_orphan(self):
        ext = ResilienceExtension()
        messages = [
            HumanMessage(content="hi"),
            _ai_with_calls(("call_lIRriSLL", "Agent")),
            # No matching ToolMessage — orphan from a crashed prior run.
            HumanMessage(content="next turn"),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": [AIMessage(content="done")]}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        repaired = captured[0]
        # Synthetic ToolMessage inserted between AIMessage and next HumanMessage.
        assert len(repaired) == 4
        assert isinstance(repaired[2], ToolMessage)
        assert repaired[2].tool_call_id == "call_lIRriSLL"
        assert repaired[2].name == "Agent"
        assert repaired[2].status == "error"
        assert repaired[2].additional_kwargs["agentkit"] == {
            "synthesized": True,
            "reason": "orphan",
        }

    async def test_reproduces_reported_failure(self):
        """The exact scenario from the production 400 error."""
        ext = ResilienceExtension()
        messages = [
            HumanMessage(content="user question"),
            _ai_with_calls(("call_lIRriSLLlwDOztGc7VwqnUag", "Agent")),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        # Pairing invariant now holds — LLM request will not 400.
        ai = captured[0][1]
        follow = captured[0][2]
        assert ai.tool_calls[0]["id"] == follow.tool_call_id

    async def test_repairs_partial_parallel_calls(self):
        """AIMessage with 3 tool_calls, only 2 ToolMessages — synthesize the missing one."""
        ext = ResilienceExtension()
        messages = [
            _ai_with_calls(("c1", "a"), ("c2", "b"), ("c3", "c")),
            ToolMessage(content="r1", tool_call_id="c1", name="a"),
            ToolMessage(content="r3", tool_call_id="c3", name="c"),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        repaired = captured[0]
        ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
        assert set(ids) == {"c1", "c2", "c3"}

        synthesized = [
            m
            for m in repaired
            if isinstance(m, ToolMessage)
            and m.additional_kwargs.get("agentkit", {}).get("synthesized")
        ]
        assert len(synthesized) == 1
        assert synthesized[0].tool_call_id == "c2"

    async def test_leaves_future_paired_tool_messages_alone(self):
        """Don't synthesize a repair for a tool_call_id whose ToolMessage
        appears later in the list (HistoryExtension can reorder blocks)."""
        ext = ResilienceExtension()
        messages = [
            _ai_with_calls(("c1", "a")),
            HumanMessage(content="noise between AI and Tool"),
            ToolMessage(content="r1", tool_call_id="c1", name="a"),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        repaired = captured[0]
        synthesized = [
            m
            for m in repaired
            if isinstance(m, ToolMessage)
            and m.additional_kwargs.get("agentkit", {}).get("synthesized")
        ]
        assert synthesized == []

    async def test_repair_disabled_passes_through(self):
        ext = ResilienceExtension(repair_orphan_tool_calls=False)
        messages = [_ai_with_calls(("c1", "a"))]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        assert captured[0] == messages  # unrepaired

    async def test_custom_repair_message(self):
        ext = ResilienceExtension(orphan_repair_message="CUSTOM")
        messages = [_ai_with_calls(("c1", "a"))]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        assert captured[0][-1].content == "CUSTOM"

    async def test_empty_messages_passthrough(self):
        ext = ResilienceExtension()
        captured: list = []

        async def handler(state):
            captured.append(state)
            return {"messages": []}

        await ext.wrap_model(state={"messages": []}, handler=handler, runtime=None)

        assert captured[0] == {"messages": []}

    async def test_multiple_orphan_ai_messages(self):
        """Orphans can accumulate across turns — all must be repaired."""
        ext = ResilienceExtension()
        messages = [
            _ai_with_calls(("c1", "a")),
            _ai_with_calls(("c2", "b")),
        ]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        repaired = captured[0]
        ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
        assert set(ids) == {"c1", "c2"}

    async def test_repair_event_callback(self):
        events: list[OrphanRepairEvent] = []
        ext = ResilienceExtension(on_orphan_repaired=events.append)
        messages = [_ai_with_calls(("c1", "a"), ("c2", "b"))]

        async def handler(state):
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        assert [e.tool_call_id for e in events] == ["c1", "c2"]
        assert events[0].tool_name == "a"

    async def test_repair_callback_failure_does_not_break_execution(self):
        def broken(e: OrphanRepairEvent) -> None:
            raise RuntimeError("sink exploded")

        ext = ResilienceExtension(on_orphan_repaired=broken)
        messages = [_ai_with_calls(("c1", "a"))]
        captured: list = []

        async def handler(state):
            captured.append(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)

        # Repair still applied; handler still invoked.
        assert len(captured[0]) == 2
