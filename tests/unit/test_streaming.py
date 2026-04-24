"""Tests for outbound tool-result stream suppression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.streaming import (
    FilteredGraph,
    StreamingFilter,
    wrap_if_filtering,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _tool(name: str) -> StructuredTool:
    def _fn(x: str) -> str:
        return x

    return StructuredTool.from_function(func=_fn, name=name, description="stub")


class _StubExt(Extension):
    def __init__(
        self,
        tools: list[Any] | None = None,
        override: dict[str, bool] | None = None,
    ) -> None:
        self._tools = tools or []
        self._override = override or {}

    @property
    def tools(self) -> list[Any]:
        return self._tools

    def stream_tool_results(self, tool_name: str) -> bool | None:
        if tool_name in self._override:
            return self._override[tool_name]
        return None


# ---------------------------------------------------------------------------
# StreamingFilter — core redaction primitives
# ---------------------------------------------------------------------------


class TestStreamingFilter:
    def test_empty_set_is_inactive(self) -> None:
        sf = StreamingFilter(frozenset())
        assert sf.active is False

    def test_non_empty_set_is_active(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        assert sf.active is True

    def test_redact_tool_message_in_suppressed_set(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        msg = ToolMessage(content="1000 lines", tool_call_id="abc", name="Read")
        redacted = sf._redact_tool_message(msg)
        assert redacted is not msg
        assert redacted.content == ""
        assert redacted.name == "Read"
        assert redacted.tool_call_id == "abc"

    def test_redact_preserves_status_error(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        msg = ToolMessage(
            content="PermissionError: denied",
            tool_call_id="abc",
            name="Read",
            status="error",
        )
        redacted = sf._redact_tool_message(msg)
        assert redacted.content == ""
        assert redacted.status == "error"
        assert redacted.name == "Read"
        assert redacted.tool_call_id == "abc"

    def test_redact_leaves_unsuppressed_tool_untouched(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        msg = ToolMessage(content="short", tool_call_id="abc", name="Glob")
        redacted = sf._redact_tool_message(msg)
        assert redacted is msg

    def test_redact_leaves_non_tool_messages_untouched(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        assert sf._redact_tool_message(AIMessage(content="hi")) is not None
        assert sf._redact_tool_message(HumanMessage(content="hi")).content == "hi"


class TestFilterAstreamChunk:
    def test_updates_mode_redacts_tool_node_messages(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        chunk = {
            "tools": {
                "messages": [
                    ToolMessage(content="big payload", tool_call_id="1", name="Read"),
                ],
            },
        }
        filtered = sf.filter_astream_chunk(chunk)
        assert filtered is not chunk
        assert filtered["tools"]["messages"][0].content == ""

    def test_updates_mode_keeps_other_node_updates(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        ai = AIMessage(content="thinking")
        chunk = {"agent": {"messages": [ai]}}
        filtered = sf.filter_astream_chunk(chunk)
        # AIMessage passes through — still the same list object when nothing changed.
        assert filtered is chunk

    def test_values_mode_redacts_full_state(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        chunk = {
            "messages": [
                AIMessage(content="thinking"),
                ToolMessage(content="big", tool_call_id="1", name="Read"),
            ],
            "sender": "agent",
        }
        filtered = sf.filter_astream_chunk(chunk)
        assert filtered["messages"][0].content == "thinking"
        assert filtered["messages"][1].content == ""
        assert filtered["sender"] == "agent"

    def test_tuple_subgraphs_shape_preserved(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        namespace = ("node:1",)
        inner = {"tools": {"messages": [ToolMessage(content="big", tool_call_id="1", name="Read")]}}
        chunk = (namespace, inner)
        filtered = sf.filter_astream_chunk(chunk)
        assert filtered[0] is namespace
        assert filtered[1]["tools"]["messages"][0].content == ""

    def test_unknown_shape_passes_through(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        chunk = "some raw string"
        assert sf.filter_astream_chunk(chunk) == "some raw string"

    def test_no_matches_returns_same_chunk(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        chunk = {"agent": {"messages": [AIMessage(content="hi")]}}
        assert sf.filter_astream_chunk(chunk) is chunk


class TestFilterAstreamEvent:
    def test_on_tool_start_passes_through(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        event = {
            "event": "on_tool_start",
            "name": "Read",
            "data": {"input": {"path": "x"}},
        }
        assert sf.filter_astream_event(event) is event

    def test_on_tool_end_suppressed_redacts_output_string(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        event = {
            "event": "on_tool_end",
            "name": "Read",
            "data": {"output": "huge payload"},
        }
        filtered = sf.filter_astream_event(event)
        assert filtered is not event
        assert filtered["data"]["output"] == ""
        assert filtered["name"] == "Read"

    def test_on_tool_end_suppressed_redacts_tool_message_output(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        tm = ToolMessage(content="huge", tool_call_id="1", name="Read", status="error")
        event = {
            "event": "on_tool_end",
            "name": "Read",
            "data": {"output": tm},
        }
        filtered = sf.filter_astream_event(event)
        out = filtered["data"]["output"]
        assert isinstance(out, ToolMessage)
        assert out.content == ""
        assert out.status == "error"
        assert out.tool_call_id == "1"

    def test_on_tool_end_unsuppressed_passes_through(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        event = {"event": "on_tool_end", "name": "Glob", "data": {"output": "x"}}
        assert sf.filter_astream_event(event) is event

    def test_on_tool_stream_chunk_redacted(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        event = {
            "event": "on_tool_stream",
            "name": "Read",
            "data": {"chunk": "partial"},
        }
        filtered = sf.filter_astream_event(event)
        assert filtered["data"]["chunk"] == ""

    def test_non_tool_event_passes_through(self) -> None:
        sf = StreamingFilter(frozenset({"Read"}))
        event = {"event": "on_chat_model_end", "name": "gpt", "data": {"output": "x"}}
        assert sf.filter_astream_event(event) is event


# ---------------------------------------------------------------------------
# AgentKit.suppressed_tool_names — resolution matrix
# ---------------------------------------------------------------------------


class TestSuppressedToolNamesResolution:
    def test_default_true_no_override_empty_suppressed_set(self) -> None:
        ext = _StubExt(tools=[_tool("Read"), _tool("Glob")])
        kit = AgentKit(extensions=[ext], stream_tool_results=True)
        assert kit.suppressed_tool_names() == frozenset()

    def test_default_false_all_tools_suppressed(self) -> None:
        ext = _StubExt(tools=[_tool("Read"), _tool("Glob")])
        kit = AgentKit(extensions=[ext], stream_tool_results=False)
        assert kit.suppressed_tool_names() == frozenset({"Read", "Glob"})

    def test_extension_override_false_against_default_true(self) -> None:
        ext = _StubExt(tools=[_tool("Read"), _tool("Glob")], override={"Read": False})
        kit = AgentKit(extensions=[ext], stream_tool_results=True)
        assert kit.suppressed_tool_names() == frozenset({"Read"})

    def test_extension_override_true_against_default_false(self) -> None:
        ext = _StubExt(tools=[_tool("Read"), _tool("Glob")], override={"Glob": True})
        kit = AgentKit(extensions=[ext], stream_tool_results=False)
        assert kit.suppressed_tool_names() == frozenset({"Read"})

    def test_first_extension_to_opine_wins(self) -> None:
        ext1 = _StubExt(tools=[_tool("Read")], override={"Read": True})
        ext2 = _StubExt(override={"Read": False})
        kit = AgentKit(extensions=[ext1, ext2], stream_tool_results=False)
        # ext1 comes first, says True → stream it → not suppressed
        assert kit.suppressed_tool_names() == frozenset()

    def test_none_override_inherits_kit_default(self) -> None:
        class _NoneExt(Extension):
            @property
            def tools(self) -> list[Any]:
                return [_tool("Read")]

            def stream_tool_results(self, tool_name: str) -> bool | None:
                return None

        kit = AgentKit(extensions=[_NoneExt()], stream_tool_results=False)
        assert kit.suppressed_tool_names() == frozenset({"Read"})


# ---------------------------------------------------------------------------
# FilteredGraph — delegation semantics
# ---------------------------------------------------------------------------


class _StubGraph:
    def __init__(self, chunks: list[Any] | None = None, events: list[Any] | None = None) -> None:
        self._chunks = chunks or []
        self._events = events or []
        self.ainvoke_calls: list[tuple[Any, Any]] = []
        self.custom_attr = "hello"

    async def ainvoke(self, inp: Any, config: Any = None) -> Any:
        self.ainvoke_calls.append((inp, config))
        return {"ok": True}

    def invoke(self, inp: Any, config: Any = None) -> Any:
        return {"ok": True, "sync": True}

    async def astream(self, inp: Any, config: Any = None) -> AsyncIterator[Any]:
        for c in self._chunks:
            yield c

    async def astream_events(
        self, inp: Any, config: Any = None, version: str = "v2"
    ) -> AsyncIterator[Any]:
        for e in self._events:
            yield e


class TestFilteredGraphDelegation:
    @pytest.mark.asyncio
    async def test_ainvoke_passes_through(self) -> None:
        graph = _StubGraph()
        wrapped = FilteredGraph(graph, StreamingFilter(frozenset({"Read"})))
        result = await wrapped.ainvoke({"in": 1}, {"cfg": 2})
        assert result == {"ok": True}
        assert graph.ainvoke_calls == [({"in": 1}, {"cfg": 2})]

    def test_getattr_forwards_to_underlying(self) -> None:
        graph = _StubGraph()
        wrapped = FilteredGraph(graph, StreamingFilter(frozenset({"Read"})))
        assert wrapped.custom_attr == "hello"

    def test_graph_property_exposes_underlying(self) -> None:
        graph = _StubGraph()
        wrapped = FilteredGraph(graph, StreamingFilter(frozenset({"Read"})))
        assert wrapped.graph is graph

    @pytest.mark.asyncio
    async def test_astream_redacts_tool_messages(self) -> None:
        chunks = [
            {"tools": {"messages": [ToolMessage(content="big", tool_call_id="1", name="Read")]}},
            {"agent": {"messages": [AIMessage(content="thinking")]}},
        ]
        graph = _StubGraph(chunks=chunks)
        wrapped = FilteredGraph(graph, StreamingFilter(frozenset({"Read"})))
        collected = [c async for c in wrapped.astream({}, None)]
        assert collected[0]["tools"]["messages"][0].content == ""
        assert collected[1]["agent"]["messages"][0].content == "thinking"

    @pytest.mark.asyncio
    async def test_astream_events_filters_tool_end(self) -> None:
        events = [
            {"event": "on_tool_start", "name": "Read", "data": {"input": {}}},
            {"event": "on_tool_end", "name": "Read", "data": {"output": "huge"}},
            {"event": "on_tool_end", "name": "Glob", "data": {"output": "short"}},
        ]
        graph = _StubGraph(events=events)
        wrapped = FilteredGraph(graph, StreamingFilter(frozenset({"Read"})))
        collected = [e async for e in wrapped.astream_events({}, None)]
        assert collected[0]["data"]["input"] == {}  # on_tool_start untouched
        assert collected[1]["data"]["output"] == ""  # suppressed
        assert collected[2]["data"]["output"] == "short"  # Glob unaffected


# ---------------------------------------------------------------------------
# wrap_if_filtering — gating
# ---------------------------------------------------------------------------


class TestWrapIfFiltering:
    def test_empty_suppressed_returns_raw_graph(self) -> None:
        graph = _StubGraph()
        assert wrap_if_filtering(graph, frozenset()) is graph

    def test_non_empty_suppressed_returns_filtered_graph(self) -> None:
        graph = _StubGraph()
        wrapped = wrap_if_filtering(graph, frozenset({"Read"}))
        assert isinstance(wrapped, FilteredGraph)
        assert wrapped.graph is graph


# ---------------------------------------------------------------------------
# Extension base default
# ---------------------------------------------------------------------------


class TestExtensionDefaultHook:
    def test_base_extension_returns_none(self) -> None:
        class _Bare(Extension):
            pass

        assert _Bare().stream_tool_results("anything") is None
