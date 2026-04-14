"""Tests for ContextCompactionExtension."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.extensions.context_compaction import (
    ContextCompactionExtension,
)
from langchain_agentkit.extensions.context_compaction.extension import (
    EVICTED_MARKER,
    redact_old_tool_messages,
)


def _tool_msg(content: str, call_id: str, name: str = "read_file") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


class TestConstruction:
    def test_default_keep_recent(self) -> None:
        ext = ContextCompactionExtension()
        assert ext.keep_recent == 5
        assert ext.tools == []
        assert ext.state_schema is None

    def test_custom_keep_recent(self) -> None:
        ext = ContextCompactionExtension(keep_recent=2)
        assert ext.keep_recent == 2

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="keep_recent must be >= 1"):
            ContextCompactionExtension(keep_recent=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="keep_recent must be >= 1"):
            ContextCompactionExtension(keep_recent=-1)

    def test_no_dependencies(self) -> None:
        assert ContextCompactionExtension().dependencies() == []


class TestPromptReminder:
    def test_returns_reminder_channel_only(self) -> None:
        ext = ContextCompactionExtension(keep_recent=3)
        out = ext.prompt({}, None)
        assert isinstance(out, dict)
        assert "reminder" in out
        assert "prompt" not in out

    def test_reminder_mentions_configured_n(self) -> None:
        ext = ContextCompactionExtension(keep_recent=7)
        reminder = ext.prompt({}, None)["reminder"]
        assert "7" in reminder


class TestRedactionLogic:
    def test_returns_same_list_when_under_threshold(self) -> None:
        msgs: list[Any] = [
            HumanMessage(content="hi"),
            _tool_msg("a", "c1"),
            _tool_msg("b", "c2"),
        ]
        out = redact_old_tool_messages(msgs, keep_recent=5)
        assert out is msgs

    def test_returns_same_list_when_exactly_at_threshold(self) -> None:
        msgs: list[Any] = [_tool_msg("a", "c1"), _tool_msg("b", "c2")]
        out = redact_old_tool_messages(msgs, keep_recent=2)
        assert out is msgs

    def test_redacts_oldest_when_over_threshold(self) -> None:
        msgs: list[Any] = [
            _tool_msg("old-a", "c1"),
            _tool_msg("old-b", "c2"),
            _tool_msg("recent-a", "c3"),
            _tool_msg("recent-b", "c4"),
        ]
        out = redact_old_tool_messages(msgs, keep_recent=2)
        assert out[0].content == EVICTED_MARKER
        assert out[1].content == EVICTED_MARKER
        assert out[2].content == "recent-a"
        assert out[3].content == "recent-b"

    def test_preserves_tool_call_id_and_name(self) -> None:
        msgs: list[Any] = [
            _tool_msg("old", "call_123", name="grep"),
            _tool_msg("r1", "call_456"),
            _tool_msg("r2", "call_789"),
        ]
        out = redact_old_tool_messages(msgs, keep_recent=2)
        assert out[0].tool_call_id == "call_123"
        assert out[0].name == "grep"
        assert out[0].content == EVICTED_MARKER

    def test_does_not_touch_non_tool_messages(self) -> None:
        ai = AIMessage(content="thinking")
        human = HumanMessage(content="do the thing")
        msgs: list[Any] = [
            human,
            _tool_msg("old", "c1"),
            ai,
            _tool_msg("r1", "c2"),
            _tool_msg("r2", "c3"),
        ]
        out = redact_old_tool_messages(msgs, keep_recent=2)
        assert out[0] is human
        assert out[2] is ai
        assert out[1].content == EVICTED_MARKER
        assert out[3].content == "r1"
        assert out[4].content == "r2"

    def test_idempotent_on_already_redacted(self) -> None:
        msgs: list[Any] = [
            _tool_msg(EVICTED_MARKER, "c1"),
            _tool_msg("r1", "c2"),
            _tool_msg("r2", "c3"),
        ]
        out = redact_old_tool_messages(msgs, keep_recent=2)
        assert out is msgs

    def test_does_not_mutate_input_list(self) -> None:
        msgs: list[Any] = [
            _tool_msg("old", "c1"),
            _tool_msg("r1", "c2"),
            _tool_msg("r2", "c3"),
        ]
        original_first = msgs[0]
        original_first_content = original_first.content
        _ = redact_old_tool_messages(msgs, keep_recent=2)
        assert msgs[0] is original_first
        assert original_first.content == original_first_content


class TestWrapModelIntegration:
    @pytest.mark.asyncio
    async def test_passes_redacted_state_to_handler(self) -> None:
        ext = ContextCompactionExtension(keep_recent=1)
        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["messages"] = state["messages"]
            return {"messages": [AIMessage(content="done")]}

        state = {
            "messages": [
                _tool_msg("old-1", "c1"),
                _tool_msg("old-2", "c2"),
                _tool_msg("recent", "c3"),
            ],
        }
        result = await ext.wrap_model(state=state, handler=handler, runtime=None)

        assert captured["messages"][0].content == EVICTED_MARKER
        assert captured["messages"][1].content == EVICTED_MARKER
        assert captured["messages"][2].content == "recent"
        assert result == {"messages": [AIMessage(content="done")]}

    @pytest.mark.asyncio
    async def test_does_not_mutate_caller_state(self) -> None:
        ext = ContextCompactionExtension(keep_recent=1)

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        original_first = _tool_msg("old", "c1")
        state = {"messages": [original_first, _tool_msg("recent", "c2")]}
        await ext.wrap_model(state=state, handler=handler, runtime=None)

        assert state["messages"][0] is original_first
        assert state["messages"][0].content == "old"

    @pytest.mark.asyncio
    async def test_passthrough_when_no_eviction_needed(self) -> None:
        ext = ContextCompactionExtension(keep_recent=5)
        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["state"] = state
            return {}

        state = {"messages": [_tool_msg("a", "c1")], "other": "value"}
        await ext.wrap_model(state=state, handler=handler, runtime=None)

        assert captured["state"] is state

    @pytest.mark.asyncio
    async def test_empty_messages(self) -> None:
        ext = ContextCompactionExtension(keep_recent=3)

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            return {"messages": []}

        result = await ext.wrap_model(state={"messages": []}, handler=handler, runtime=None)
        assert result == {"messages": []}

    @pytest.mark.asyncio
    async def test_hook_is_registered(self) -> None:
        ext = ContextCompactionExtension()
        hooks = ext.get_all_hooks()
        assert ("wrap", "model") in hooks
        assert len(hooks[("wrap", "model")]) == 1
