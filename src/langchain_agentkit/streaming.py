"""Outbound-stream payload suppression for tool-result messages.

When a compiled graph is streamed to a client, every ``ToolMessage`` the
graph produces appears in the outbound stream. The payload can be huge
(file reads, grep output, nested-agent traces) and is often pure noise
for the client — the LLM still needs it in state, but the UI doesn't.

This module implements a thin wrapper that redacts ``ToolMessage.content``
and ``on_tool_end`` / ``on_tool_stream`` ``data.output`` for a configured
set of tool names, while leaving the envelope (``name``, ``tool_call_id``,
``status``) and every other stream surface untouched.

The wrapper layers **on top of** LangGraph primitives — it never invents
new stream semantics, never mutates state, and never touches the
checkpointer. Tool results persist in full; only the outbound stream is
shaped.

Resolution: :meth:`AgentKit.suppressed_tool_names` produces the set of
tool names whose payload should be redacted, combining the kit-level
default (``AgentKit(stream_tool_results=...)``) with per-tool overrides
contributed by extensions (``Extension.stream_tool_results(tool_name)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["FilteredGraph", "StreamingFilter", "wrap_if_filtering"]


_REDACTED = ""


class StreamingFilter:
    """Redacts tool-result payloads in outbound stream chunks and events.

    Holds the frozen set of tool names whose ``ToolMessage.content`` and
    ``on_tool_*`` ``data.output`` should be replaced with an empty string
    on the outbound stream.  A filter with an empty set is a no-op and
    callers should avoid wrapping the graph at all in that case.
    """

    __slots__ = ("suppressed",)

    def __init__(self, suppressed: frozenset[str]) -> None:
        self.suppressed = suppressed

    @property
    def active(self) -> bool:
        """True when at least one tool name is suppressed."""
        return bool(self.suppressed)

    def _redact_tool_message(self, msg: Any) -> Any:
        """Return a ToolMessage copy with redacted content, or ``msg`` unchanged."""
        if not isinstance(msg, ToolMessage):
            return msg
        if msg.name not in self.suppressed:
            return msg
        return msg.model_copy(update={"content": _REDACTED})

    def _redact_messages(self, messages: Any) -> Any:
        """Return a new list with suppressed-tool ToolMessages redacted."""
        if not isinstance(messages, list):
            return messages
        redacted: list[Any] = []
        changed = False
        for m in messages:
            rm = self._redact_tool_message(m)
            if rm is not m:
                changed = True
            redacted.append(rm)
        return redacted if changed else messages

    def _redact_state_update(self, update: Any) -> Any:
        """Redact ``messages`` inside a state-update dict, if present."""
        if not isinstance(update, dict) or "messages" not in update:
            return update
        new_messages = self._redact_messages(update["messages"])
        if new_messages is update["messages"]:
            return update
        new_update = dict(update)
        new_update["messages"] = new_messages
        return new_update

    def filter_astream_chunk(self, chunk: Any) -> Any:
        """Redact tool-result payloads in an ``astream`` chunk.

        Handles the shapes LangGraph emits:

        - ``stream_mode="updates"`` — ``{node_name: state_update}``.
        - ``stream_mode="values"`` — the full state dict.
        - Multi-mode — ``(mode, chunk)`` tuple.
        - ``subgraphs=True`` — ``(namespace, chunk)`` or
          ``(namespace, mode, chunk)``.
        - ``stream_mode="messages"`` — ``(message_chunk, metadata)`` tuple
          with no ``ToolMessage`` (already filtered by LangGraph).
        - ``"custom"`` / ``"debug"`` — pass through unchanged.

        Pass-through is the safe default — we only rewrite shapes we can
        confidently recognize as state-carrying.
        """
        if isinstance(chunk, tuple):
            # Recurse into the last element (the actual payload) and
            # rebuild the tuple. Prefix elements are routing metadata.
            filtered_last = self.filter_astream_chunk(chunk[-1])
            if filtered_last is chunk[-1]:
                return chunk
            return (*chunk[:-1], filtered_last)

        if not isinstance(chunk, dict):
            return chunk

        # ``values`` mode: chunk is the state dict itself.
        if "messages" in chunk:
            return self._redact_state_update(chunk)

        # ``updates`` mode: chunk is {node_name: state_update}. Redact
        # each value that looks like a state update with messages.
        changed = False
        new_chunk: dict[str, Any] = {}
        for node_name, value in chunk.items():
            redacted_value = self._redact_state_update(value)
            if redacted_value is not value:
                changed = True
            new_chunk[node_name] = redacted_value
        return new_chunk if changed else chunk

    def filter_astream_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Redact ``data.output`` / ``data.chunk`` on ``on_tool_*`` events.

        ``on_tool_start`` passes through — clients need the lifecycle
        signal to render "tool running". Only the payload-carrying
        events (``on_tool_end``, ``on_tool_stream``) are rewritten.
        """
        kind = event.get("event")
        if kind not in ("on_tool_end", "on_tool_stream"):
            return event
        name = event.get("name")
        if name not in self.suppressed:
            return event
        data = event.get("data")
        if not isinstance(data, dict):
            return event
        new_data = dict(data)
        if kind == "on_tool_end":
            output = data.get("output")
            new_data["output"] = self._redact_event_output(output)
        else:  # on_tool_stream
            if "chunk" in data:
                new_data["chunk"] = self._redact_event_output(data.get("chunk"))
        new_event = dict(event)
        new_event["data"] = new_data
        return new_event

    def _redact_event_output(self, output: Any) -> Any:
        """Redact a tool event's output payload preserving envelope fields."""
        if isinstance(output, ToolMessage):
            return output.model_copy(update={"content": _REDACTED})
        if isinstance(output, str):
            return _REDACTED
        if isinstance(output, dict) and "messages" in output:
            return self._redact_state_update(output)
        # Unknown shape — elide to empty string so payload never crosses the wire.
        return _REDACTED


class FilteredGraph:
    """Delegating proxy that filters tool-result payloads on the outbound stream.

    Wraps a compiled LangGraph runnable. ``astream`` and ``astream_events``
    are intercepted and piped through a :class:`StreamingFilter`; every
    other attribute (``ainvoke``, ``invoke``, ``get_graph``, ``get_state``,
    checkpointer access, etc.) forwards to the wrapped graph unchanged.

    Use :func:`wrap_if_filtering` to construct — it returns the original
    graph when no suppression is configured, avoiding an unnecessary
    wrapper layer.
    """

    __slots__ = ("_graph", "_filter")

    def __init__(self, graph: Any, stream_filter: StreamingFilter) -> None:
        self._graph = graph
        self._filter = stream_filter

    @property
    def graph(self) -> Any:
        """The underlying compiled graph, bypassing the stream filter."""
        return self._graph

    @property
    def stream_filter(self) -> StreamingFilter:
        """The active streaming filter."""
        return self._filter

    def __getattr__(self, item: str) -> Any:
        # Forward anything not explicitly overridden to the wrapped graph.
        return getattr(self._graph, item)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await self._graph.ainvoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._graph.invoke(*args, **kwargs)

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for chunk in self._graph.astream(*args, **kwargs):
            yield self._filter.filter_astream_chunk(chunk)

    async def astream_events(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        async for event in self._graph.astream_events(*args, **kwargs):
            yield self._filter.filter_astream_event(event)


def wrap_if_filtering(graph: Any, suppressed: frozenset[str]) -> Any:
    """Wrap ``graph`` in :class:`FilteredGraph` only when there is work to do.

    An empty suppressed set means every tool streams normally — no wrapping
    needed, and returning the raw graph keeps the public API surface
    identical to pre-feature behavior for callers that don't configure
    suppression.
    """
    if not suppressed:
        return graph
    return FilteredGraph(graph, StreamingFilter(suppressed))
