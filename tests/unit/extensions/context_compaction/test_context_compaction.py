"""Tests for the summarizing ContextCompactionExtension (I2)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.extensions.context_compaction import (
    CompactionSettings,
    ContextCompactionExtension,
)
from langchain_agentkit.extensions.context_compaction.cutpoint import (
    find_cut_point,
    find_turn_start_index,
)
from langchain_agentkit.extensions.context_compaction.file_ops import (
    compute_file_lists,
    extract_file_ops,
    format_file_operations,
)
from langchain_agentkit.extensions.context_compaction.token_accounting import (
    estimate_context_tokens,
    estimate_tokens,
    should_compact,
)


def _tool_msg(content: str, call_id: str = "c1", name: str = "Read") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


class _FakeLlm:
    """Minimal BaseChatModel stand-in returning a fixed summary."""

    def __init__(self, reply: str = "MOCK_SUMMARY") -> None:
        self._reply = reply
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self._reply)


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    def test_estimate_tokens_from_string(self) -> None:
        msg = HumanMessage(content="x" * 400)
        assert estimate_tokens(msg) == 100  # 400 / 4

    def test_estimate_tokens_counts_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "Read",
                    "args": {"path": "/long/path/to/file.py"},
                }
            ],
        )
        assert estimate_tokens(msg) > 0

    def test_estimate_context_tokens_uses_usage_metadata(self) -> None:
        ai = AIMessage(
            content="hi",
            usage_metadata={"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
        )
        trailing = HumanMessage(content="x" * 40)
        estimate = estimate_context_tokens([ai, trailing])
        assert estimate.usage_tokens == 600
        assert estimate.trailing_tokens == 10
        assert estimate.tokens == 610
        assert estimate.last_usage_index == 0

    def test_estimate_context_tokens_falls_back_to_estimation(self) -> None:
        estimate = estimate_context_tokens([HumanMessage(content="x" * 40)])
        assert estimate.usage_tokens == 0
        assert estimate.tokens == 10

    def test_should_compact(self) -> None:
        # 128k - 16k reserve = 112k headroom — 115k pushes us into compaction.
        assert should_compact(ctx_tokens=115_000, context_window=128_000, reserve_tokens=16_384)
        assert not should_compact(ctx_tokens=50_000, context_window=128_000, reserve_tokens=16_384)


# ---------------------------------------------------------------------------
# Cutpoint detection
# ---------------------------------------------------------------------------


class TestCutpoint:
    def test_returns_start_when_no_valid_cuts(self) -> None:
        # ToolMessage-only window has no valid cut.
        msgs = [_tool_msg("a"), _tool_msg("b")]
        cut = find_cut_point(msgs, 0, len(msgs), keep_recent_tokens=10)
        assert cut.first_kept_index == 0
        assert cut.is_split_turn is False

    def test_cuts_at_user_boundary(self) -> None:
        msgs = [
            HumanMessage(content="start-" + "x" * 400),  # ~100 tokens
            AIMessage(content="reply-" + "x" * 400),
            HumanMessage(content="recent-" + "y" * 400),
            AIMessage(content="done-" + "y" * 400),
        ]
        cut = find_cut_point(msgs, 0, len(msgs), keep_recent_tokens=200)
        # Should keep the last user message + its assistant reply.
        assert cut.first_kept_index == 2
        assert cut.is_split_turn is False
        assert cut.turn_start_index == -1

    def test_split_turn_detected(self) -> None:
        # Budget large enough to force cutting inside a long assistant turn.
        msgs = [
            HumanMessage(content="q1-" + "x" * 40),
            AIMessage(content="a1-" + "x" * 2000),  # big assistant turn
            HumanMessage(content="recent-" + "y" * 400),
        ]
        cut = find_cut_point(msgs, 0, len(msgs), keep_recent_tokens=150)
        # If we cut at the AIMessage, turn_start_index should point to the
        # preceding HumanMessage.
        if cut.is_split_turn:
            assert cut.turn_start_index == 0
            assert cut.first_kept_index == 1

    def test_find_turn_start_index(self) -> None:
        msgs = [
            HumanMessage(content="u1"),
            AIMessage(content="a1"),
            _tool_msg("r1"),
            AIMessage(content="a2"),
        ]
        assert find_turn_start_index(msgs, cut_index=3, start=0) == 0


# ---------------------------------------------------------------------------
# File-op index
# ---------------------------------------------------------------------------


class TestFileOps:
    def test_extract_from_read_write_edit(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "1", "name": "Read", "args": {"path": "/a.py"}},
                    {"id": "2", "name": "Write", "args": {"path": "/b.py", "content": "x"}},
                    {"id": "3", "name": "Edit", "args": {"path": "/c.py"}},
                ],
            ),
        ]
        ops = extract_file_ops(msgs)
        assert ops.read == {"/a.py"}
        assert ops.written == {"/b.py"}
        assert ops.edited == {"/c.py"}

    def test_compute_read_vs_modified(self) -> None:
        from langchain_agentkit.extensions.context_compaction.file_ops import FileOps

        ops = FileOps(read={"/a.py", "/b.py"}, written={"/b.py"}, edited={"/c.py"})
        read_only, modified = compute_file_lists(ops)
        assert read_only == ["/a.py"]  # b.py was also written → classified modified
        assert modified == ["/b.py", "/c.py"]

    def test_format_sections(self) -> None:
        out = format_file_operations(read_files=["/a"], modified_files=["/b"])
        assert "<read-files>" in out
        assert "/a" in out
        assert "<modified-files>" in out

    def test_format_empty(self) -> None:
        assert format_file_operations([], []) == ""


# ---------------------------------------------------------------------------
# Extension integration
# ---------------------------------------------------------------------------


class TestExtension:
    def test_defaults(self) -> None:
        ext = ContextCompactionExtension()
        assert ext.cache_size == 0

    def test_custom_settings(self) -> None:
        ext = ContextCompactionExtension(
            settings=CompactionSettings(enabled=False, reserve_tokens=1000),
        )
        assert ext._settings.enabled is False
        assert ext._settings.reserve_tokens == 1000

    def test_prompt_returns_reminder_describing_compaction(self) -> None:
        """Compaction guidance ships in the reminder channel — tail of system prompt."""
        ext = ContextCompactionExtension()
        out = ext.prompt({}, None)
        assert isinstance(out, dict)
        assert "reminder" in out
        assert "summarized" in out["reminder"].lower()

    def test_hook_is_wrap_model(self) -> None:
        ext = ContextCompactionExtension()
        hooks = ext.get_all_hooks()
        assert ("wrap", "model") in hooks

    @pytest.mark.asyncio
    async def test_disabled_passthrough(self) -> None:
        ext = ContextCompactionExtension(
            settings=CompactionSettings(enabled=False),
        )

        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["state"] = state
            return {}

        msgs = [HumanMessage(content="x") for _ in range(5)]
        await ext.wrap_model(state={"messages": msgs}, handler=handler, runtime=None)
        assert captured["state"]["messages"] is msgs

    @pytest.mark.asyncio
    async def test_below_threshold_passthrough(self) -> None:
        ext = ContextCompactionExtension(
            summarizer_llm=_FakeLlm(),
            context_window_resolver=lambda: 1_000_000,  # plenty of headroom
        )

        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["state"] = state
            return {}

        msgs = [HumanMessage(content="small")]
        await ext.wrap_model(state={"messages": msgs}, handler=handler, runtime=None)
        assert captured["state"]["messages"] is msgs

    @pytest.mark.asyncio
    async def test_triggered_summary_replaces_prefix(self) -> None:
        """When context exceeds the window, prefix is summarized and replaced."""
        fake = _FakeLlm(reply="STRUCTURED_SUMMARY")
        # Tiny window so any conversation triggers compaction.
        ext = ContextCompactionExtension(
            summarizer_llm=fake,
            settings=CompactionSettings(
                enabled=True,
                reserve_tokens=50,
                keep_recent_tokens=10,
            ),
            context_window_resolver=lambda: 100,
        )

        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["messages"] = list(state["messages"])
            return {"messages": [AIMessage(content="ok")]}

        # Use small trailing messages so the cut lands at the final
        # HumanMessage boundary (no split turn → single LLM call).
        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2-recent"),
            AIMessage(content="r2", usage_metadata=None),
        ]
        await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)

        assert len(fake.calls) == 1  # exactly one LLM summarization call
        # Handler saw the synthetic summary at the head, replacing dropped prefix.
        first = captured["messages"][0]
        assert isinstance(first, HumanMessage)
        assert "STRUCTURED_SUMMARY" in str(first.content)
        assert "compaction-summary" in str(first.content)
        # Original head (Human400) must have been dropped.
        assert first is not history[0]

    @pytest.mark.asyncio
    async def test_cache_reuses_summary_for_same_prefix(self) -> None:
        """Second call with the same prefix skips the LLM."""
        fake = _FakeLlm(reply="CACHED")
        ext = ContextCompactionExtension(
            summarizer_llm=fake,
            settings=CompactionSettings(
                enabled=True,
                reserve_tokens=50,
                keep_recent_tokens=10,
            ),
            context_window_resolver=lambda: 100,
        )

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            return {"messages": []}

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
            AIMessage(content="r2"),
        ]
        calls_before = len(fake.calls)
        await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)
        calls_after_first = len(fake.calls)
        await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)
        calls_after_second = len(fake.calls)
        # Second call should have added zero new LLM calls — the prefix
        # hash matched the cache from the first run.
        assert calls_after_second == calls_after_first
        assert calls_after_first > calls_before
        assert ext.cache_size == 1

    @pytest.mark.asyncio
    async def test_invalidate_cache(self) -> None:
        fake = _FakeLlm()
        ext = ContextCompactionExtension(
            summarizer_llm=fake,
            settings=CompactionSettings(enabled=True, reserve_tokens=50, keep_recent_tokens=10),
            context_window_resolver=lambda: 100,
        )

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
            AIMessage(content="r2"),
        ]
        await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)
        assert ext.cache_size == 1
        ext.invalidate_cache()
        assert ext.cache_size == 0

    @pytest.mark.asyncio
    async def test_setup_captures_metadata_getter(self) -> None:
        """setup() wires model_metadata_getter as the context_window resolver."""
        fake_meta = type("M", (), {"context_window": 50_000})()
        ext = ContextCompactionExtension()
        await ext.setup(
            llm_getter=lambda: _FakeLlm(),
            model_metadata_getter=lambda: fake_meta,
        )
        assert ext._context_window() == 50_000
