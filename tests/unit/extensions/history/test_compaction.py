"""Tests for CompactionStrategy and its private helpers."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_agentkit.extensions.history import (
    CompactionStrategy,
    HistoryExtension,
)
from langchain_agentkit.extensions.history._file_ops import (
    FileOps,
    compute_file_lists,
    extract_file_ops,
    format_file_operations,
)
from langchain_agentkit.extensions.history._token_accounting import (
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
# Token accounting helper
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
        assert should_compact(ctx_tokens=115_000, context_window=128_000, reserve_tokens=16_384)
        assert not should_compact(ctx_tokens=50_000, context_window=128_000, reserve_tokens=16_384)


# ---------------------------------------------------------------------------
# File-op index helper
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
# CompactionStrategy
# ---------------------------------------------------------------------------


class TestCompactionStrategy:
    @pytest.mark.asyncio
    async def test_below_threshold_passthrough(self) -> None:
        """Under the trigger threshold, returns messages unchanged."""
        strategy = CompactionStrategy(
            summarizer_llm=_FakeLlm(),
            context_window_resolver=lambda: 1_000_000,
        )
        msgs = [HumanMessage(content="small"), AIMessage(content="ok")]
        out = await strategy.transform(msgs, runtime=None)
        assert out is msgs

    @pytest.mark.asyncio
    async def test_too_few_messages_passthrough(self) -> None:
        """Single-message history can never trigger compaction."""
        strategy = CompactionStrategy(summarizer_llm=_FakeLlm())
        msgs = [HumanMessage(content="only one")]
        out = await strategy.transform(msgs, runtime=None)
        assert out is msgs

    @pytest.mark.asyncio
    async def test_triggered_collapses_to_single_summary(self) -> None:
        """When threshold exceeded, returns ONE HumanMessage with the summary."""
        fake = _FakeLlm(reply="STRUCTURED_SUMMARY")
        strategy = CompactionStrategy(
            summarizer_llm=fake,
            reserve_tokens=50,
            context_window_resolver=lambda: 100,
        )

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
            AIMessage(content="r2"),
        ]
        out = await strategy.transform(history, runtime=None)

        assert len(fake.calls) == 1
        assert len(out) == 1
        assert isinstance(out[0], HumanMessage)
        assert "STRUCTURED_SUMMARY" in str(out[0].content)
        assert "compaction-summary" in str(out[0].content)
        # None of the originals carried through.
        for original in history:
            assert original is not out[0]

    @pytest.mark.asyncio
    async def test_preserves_leading_system_message(self) -> None:
        """A leading SystemMessage is preserved verbatim, then the summary follows."""
        fake = _FakeLlm(reply="SUMMARY")
        strategy = CompactionStrategy(
            summarizer_llm=fake,
            reserve_tokens=50,
            context_window_resolver=lambda: 100,
        )
        system = SystemMessage(content="you are a helpful assistant")
        history = [
            system,
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
        ]
        out = await strategy.transform(history, runtime=None)

        assert out[0] is system
        assert len(out) == 2
        assert isinstance(out[1], HumanMessage)
        assert "SUMMARY" in str(out[1].content)

    @pytest.mark.asyncio
    async def test_last_summary_chains_into_next_compaction(self) -> None:
        """Second compaction passes the prior summary via UPDATE prompt."""
        fake = _FakeLlm(reply="SUMMARY-N")
        strategy = CompactionStrategy(
            summarizer_llm=fake,
            reserve_tokens=50,
            context_window_resolver=lambda: 100,
        )
        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
            AIMessage(content="r2"),
        ]
        await strategy.transform(history, runtime=None)
        await strategy.transform(history, runtime=None)

        # On the second call the user prompt should reference the prior summary.
        second_call_user_msg = fake.calls[1][-1]
        assert "<previous-summary>" in str(second_call_user_msg.content)

    @pytest.mark.asyncio
    async def test_setup_supplies_llm_when_not_constructor_provided(self) -> None:
        """setup(llm_getter=...) is used to resolve the summarizer LLM lazily."""
        fake = _FakeLlm(reply="SUM")
        strategy = CompactionStrategy(
            reserve_tokens=50,
            context_window_resolver=lambda: 100,
        )
        await strategy.setup(llm_getter=lambda: fake)

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
        ]
        out = await strategy.transform(history, runtime=None)
        assert len(fake.calls) == 1
        assert any("SUM" in str(m.content) for m in out if isinstance(m, HumanMessage))

    @pytest.mark.asyncio
    async def test_raises_when_no_llm_available(self) -> None:
        """Triggered compaction without any LLM source is a clear error."""
        strategy = CompactionStrategy(
            reserve_tokens=50,
            context_window_resolver=lambda: 100,
        )
        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
        ]
        with pytest.raises(RuntimeError, match="summarizer_llm"):
            await strategy.transform(history, runtime=None)


# ---------------------------------------------------------------------------
# CompactionStrategy through HistoryExtension
# ---------------------------------------------------------------------------


class TestCompactionThroughHistoryExtension:
    @pytest.mark.asyncio
    async def test_extension_persists_summary_to_state(self) -> None:
        """HistoryExtension wraps the strategy and writes ReplaceMessages."""
        from langchain_agentkit.extensions.history.state import ReplaceMessages

        fake = _FakeLlm(reply="STRUCTURED")
        ext = HistoryExtension(
            strategy=CompactionStrategy(
                summarizer_llm=fake,
                reserve_tokens=50,
                context_window_resolver=lambda: 100,
            )
        )

        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["seen"] = list(state["messages"])
            return {"messages": [AIMessage(content="response")]}

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
            HumanMessage(content="q2"),
            AIMessage(content="r2"),
        ]

        result = await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)

        # Handler saw the collapsed summary as the only history entry.
        assert len(captured["seen"]) == 1
        assert "STRUCTURED" in str(captured["seen"][0].content)

        # The graph-state write is a ReplaceMessages with summary + response.
        assert isinstance(result["messages"], ReplaceMessages)

    @pytest.mark.asyncio
    async def test_extension_setup_forwards_llm_getter(self) -> None:
        """HistoryExtension.setup() forwards llm_getter to the strategy."""
        fake = _FakeLlm(reply="OK")
        ext = HistoryExtension(
            strategy=CompactionStrategy(
                reserve_tokens=50,
                context_window_resolver=lambda: 100,
            )
        )
        await ext.setup(llm_getter=lambda: fake)

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            return {"messages": []}

        history = [
            HumanMessage(content="q1-" + "x" * 400),
            AIMessage(
                content="a1-" + "x" * 400,
                usage_metadata={"input_tokens": 150, "output_tokens": 0, "total_tokens": 150},
            ),
        ]
        await ext.wrap_model(state={"messages": history}, handler=handler, runtime=None)
        assert len(fake.calls) == 1
