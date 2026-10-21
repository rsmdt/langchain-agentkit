from __future__ import annotations

from typing import Any

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_agentkit.extensions.history import (
    CompactionStrategy,
    HandoffStrategy,
    HistoryExtension,
    ReplaceMessages,
)


class _FakeLlm:
    def __init__(self, reply: str = "BRIEF") -> None:
        self._reply = reply
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self._reply)


class _ScriptedHandler:
    def __init__(self, *outcomes: BaseException | None) -> None:
        self._outcomes = list(outcomes)
        self.seen: list[list[Any]] = []

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(list(state["messages"]))
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if outcome is not None:
            raise outcome
        return {"messages": [AIMessage(content="response")]}


def _brief_input(llm: _FakeLlm, call: int = 0) -> str:
    return "\n".join(str(m.content) for m in llm.calls[call])


def _history(tag: str) -> list[Any]:
    return [
        SystemMessage(content="persona"),
        HumanMessage(content=f"first-request-{tag}"),
        AIMessage(
            content=f"reading-{tag}",
            tool_calls=[{"id": f"r-{tag}", "name": "Read", "args": {"path": f"/{tag}/notes.txt"}}],
        ),
        ToolMessage(content="x" * 2000, tool_call_id=f"r-{tag}", name="Read"),
        AIMessage(content=f"answer-{tag}"),
        HumanMessage(content=f"latest-goal-{tag}"),
        AIMessage(
            content="",
            tool_calls=[{"id": f"w-{tag}", "name": "Write", "args": {"path": f"/{tag}/out.txt"}}],
        ),
        ToolMessage(content="y" * 4000, tool_call_id=f"w-{tag}", name="Write"),
    ]


def _strategy(llm: _FakeLlm, context_window: int = 100) -> HandoffStrategy:
    return HandoffStrategy(summarizer_llm=llm, context_window=context_window, reserve_tokens=0)


class TestHandoffCompact:
    @pytest.mark.asyncio
    async def test_latest_user_request_reaches_brief_call_as_goal(self) -> None:
        llm = _FakeLlm()
        await _strategy(llm).compact(_history("a"), runtime=None)

        assert len(llm.calls) == 1
        assert "latest-goal-a" in _brief_input(llm)

    @pytest.mark.asyncio
    async def test_tracked_files_reach_brief_call(self) -> None:
        llm = _FakeLlm()
        await _strategy(llm).compact(_history("a"), runtime=None)

        brief_input = _brief_input(llm)
        assert "/a/notes.txt" in brief_input
        assert "/a/out.txt" in brief_input

    @pytest.mark.asyncio
    async def test_result_is_system_brief_and_verbatim_goal(self) -> None:
        history = _history("a")
        result = await _strategy(_FakeLlm(reply="THE-BRIEF")).compact(history, runtime=None)

        assert len(result) == 3
        assert result[0] is history[0]
        assert isinstance(result[1], HumanMessage)
        assert "THE-BRIEF" in result[1].content
        assert result[2] is history[5]

    @pytest.mark.asyncio
    async def test_without_system_message(self) -> None:
        history = _history("a")[1:]
        result = await _strategy(_FakeLlm()).compact(history, runtime=None)

        assert len(result) == 2
        assert result[1] is history[4]

    @pytest.mark.asyncio
    async def test_prior_brief_is_input_and_replaced(self) -> None:
        llm = _FakeLlm(reply="BRIEF-ONE")
        strategy = _strategy(llm)
        first = await strategy.compact(_history("a"), runtime=None)

        llm._reply = "BRIEF-TWO"
        second_round = [*first, AIMessage(content="more"), HumanMessage(content="next-goal")]
        result = await strategy.compact(second_round, runtime=None)

        assert "BRIEF-ONE" in _brief_input(llm, 1)
        assert "next-goal" in _brief_input(llm, 1)
        assert len(result) == 3
        assert "BRIEF-TWO" in result[1].content
        assert "BRIEF-ONE" not in result[1].content
        assert result[2] is second_round[-1]

    @pytest.mark.asyncio
    async def test_reads_back_compaction_summary_as_prior_brief(self) -> None:
        compacted = await CompactionStrategy(summarizer_llm=_FakeLlm(reply="OLD-SUMMARY")).compact(
            _history("a"), runtime=None
        )
        llm = _FakeLlm()
        result = await _strategy(llm).compact(
            [*compacted, HumanMessage(content="new-goal")], runtime=None
        )

        assert "OLD-SUMMARY" in _brief_input(llm)
        assert len(result) == 3
        assert "OLD-SUMMARY" not in result[1].content

    @pytest.mark.asyncio
    async def test_conversations_through_one_instance_do_not_leak(self) -> None:
        llm = _FakeLlm()
        strategy = _strategy(llm)
        first_alpha = await strategy.compact(_history("alpha"), runtime=None)
        await strategy.compact(_history("bravo"), runtime=None)
        await strategy.compact([*first_alpha, HumanMessage(content="follow-up")], runtime=None)

        assert "alpha" not in _brief_input(llm, 1)
        assert "bravo" not in _brief_input(llm, 2)


class TestHandoffTransform:
    @pytest.mark.asyncio
    async def test_below_threshold_passes_through(self) -> None:
        llm = _FakeLlm()
        history = _history("a")
        result = await _strategy(llm, context_window=1_000_000).transform(history, runtime=None)

        assert result is history
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_above_threshold_hands_off(self) -> None:
        llm = _FakeLlm()
        history = _history("a")
        result = await _strategy(llm).transform(history, runtime=None)

        assert len(llm.calls) == 1
        assert [result[0], result[2]] == [history[0], history[5]]

    @pytest.mark.asyncio
    async def test_uses_kit_llm_when_no_dedicated_model(self) -> None:
        llm = _FakeLlm()
        strategy = HandoffStrategy(context_window=100, reserve_tokens=0)
        await strategy.setup(llm_getter=lambda: llm)  # type: ignore[arg-type,return-value]
        await strategy.transform(_history("a"), runtime=None)

        assert len(llm.calls) == 1


class TestHandoffOverflowRecovery:
    @pytest.mark.asyncio
    async def test_overflow_hands_off_and_retries(self) -> None:
        llm = _FakeLlm()
        handler = _ScriptedHandler(ContextOverflowError("too long"))
        extension = HistoryExtension(
            strategy=HandoffStrategy(summarizer_llm=llm, context_window_resolver=lambda: 1_000_000)
        )
        history = _history("a")

        result = await extension.wrap_model(
            state={"messages": history}, handler=handler, runtime=None
        )

        assert len(handler.seen) == 2
        retried = handler.seen[1]
        assert len(retried) == 3
        assert retried[0] is history[0]
        assert retried[2] is history[5]
        assert isinstance(result["messages"], ReplaceMessages)
        persisted = [m for m in result["messages"] if m.type != "remove"]
        assert persisted[:3] == retried
        assert persisted[3].content == "response"
