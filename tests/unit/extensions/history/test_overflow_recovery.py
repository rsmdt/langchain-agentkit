from __future__ import annotations

from typing import Any

import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.extensions.history import (
    CompactionStrategy,
    CountStrategy,
    HistoryExtension,
    ReplaceMessages,
)


class _FakeLlm:
    def __init__(self, reply: str = "SUMMARY") -> None:
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


def _large_history() -> list[Any]:
    return [
        HumanMessage(content="q1-" + "x" * 4000),
        AIMessage(content="a1-" + "x" * 4000),
        HumanMessage(content="q2"),
    ]


def _compacting_extension(summarizer: _FakeLlm) -> HistoryExtension:
    return HistoryExtension(
        strategy=CompactionStrategy(
            summarizer_llm=summarizer,
            context_window_resolver=lambda: 1_000_000,
        )
    )


class TestContextOverflowRecovery:
    @pytest.mark.asyncio
    async def test_overflow_compacts_and_retries_once(self) -> None:
        summarizer = _FakeLlm()
        handler = _ScriptedHandler(ContextOverflowError("too long"))

        result = await _compacting_extension(summarizer).wrap_model(
            state={"messages": _large_history()}, handler=handler, runtime=None
        )

        assert len(handler.seen) == 2
        assert len(handler.seen[1]) == 1
        assert len(summarizer.calls) == 1
        assert isinstance(result["messages"], ReplaceMessages)
        persisted = [m for m in result["messages"] if m.type != "remove"]
        assert persisted[0] is handler.seen[1][0]
        assert persisted[1].content == "response"

    @pytest.mark.asyncio
    async def test_reraises_when_compaction_is_not_smaller(self) -> None:
        summarizer = _FakeLlm(reply="y" * 50_000)
        handler = _ScriptedHandler(ContextOverflowError("too long"))

        with pytest.raises(ContextOverflowError):
            await _compacting_extension(summarizer).wrap_model(
                state={"messages": _large_history()}, handler=handler, runtime=None
            )
        assert len(handler.seen) == 1

    @pytest.mark.asyncio
    async def test_second_overflow_propagates(self) -> None:
        handler = _ScriptedHandler(
            ContextOverflowError("too long"), ContextOverflowError("still too long")
        )

        with pytest.raises(ContextOverflowError, match="still too long"):
            await _compacting_extension(_FakeLlm()).wrap_model(
                state={"messages": _large_history()}, handler=handler, runtime=None
            )
        assert len(handler.seen) == 2

    @pytest.mark.asyncio
    async def test_other_errors_propagate_without_compaction(self) -> None:
        summarizer = _FakeLlm()
        handler = _ScriptedHandler(ValueError("boom"))

        with pytest.raises(ValueError, match="boom"):
            await _compacting_extension(summarizer).wrap_model(
                state={"messages": _large_history()}, handler=handler, runtime=None
            )
        assert len(handler.seen) == 1
        assert summarizer.calls == []

    @pytest.mark.asyncio
    async def test_strategy_without_compaction_reraises(self) -> None:
        handler = _ScriptedHandler(ContextOverflowError("too long"))
        extension = HistoryExtension(strategy=CountStrategy(max_messages=50))

        with pytest.raises(ContextOverflowError):
            await extension.wrap_model(
                state={"messages": _large_history()}, handler=handler, runtime=None
            )
        assert len(handler.seen) == 1
