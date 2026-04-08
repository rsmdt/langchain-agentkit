"""Tests for HistoryExtension and history strategies."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_agentkit.extensions.history import (
    HistoryExtension,
    HistoryStrategy,
)
from langchain_agentkit.hook_runner import HookRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMessage:
    """Minimal message stand-in for unit tests."""

    def __init__(self, content: str) -> None:
        self.content = content

    def __repr__(self) -> str:
        return f"FakeMessage({self.content!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeMessage) and self.content == other.content


class FakeSystemMessage(FakeMessage):
    """Mimics SystemMessage (class name checked by strategies)."""


FakeSystemMessage.__name__ = "SystemMessage"
FakeSystemMessage.__qualname__ = "SystemMessage"


class IdMessage(FakeMessage):
    """Message with an id attribute, mimicking LangChain messages."""

    def __init__(self, content: str, *, msg_id: str) -> None:
        super().__init__(content)
        self.id = msg_id


def _msgs(n: int, prefix: str = "msg") -> list[FakeMessage]:
    return [FakeMessage(f"{prefix}{i}") for i in range(n)]


def _id_msgs(n: int) -> list[IdMessage]:
    return [IdMessage(f"m{i}", msg_id=f"id-{i}") for i in range(n)]


async def _run_wrap(
    ext: HistoryExtension,
    messages: list[Any],
    response: list[Any] | None = None,
) -> dict[str, Any]:
    """Run wrap_model with a fake handler that returns response messages."""
    resp = response if response is not None else [FakeMessage("response")]

    async def fake_handler(state: Any) -> dict[str, Any]:
        return {"messages": resp}

    return await ext.wrap_model(state={"messages": messages}, handler=fake_handler, runtime=None)


# ===========================================================================
# strategy="count"
# ===========================================================================


class TestCountStrategy:
    @pytest.mark.asyncio
    async def test_basic_truncation(self):
        from langchain_agentkit.extensions.history.state import ReplaceMessages

        ext = HistoryExtension(strategy="count", max_messages=3)
        result = await _run_wrap(ext, _msgs(6))
        assert isinstance(result["messages"], ReplaceMessages)

    @pytest.mark.asyncio
    async def test_handler_receives_truncated_messages(self):
        """The inner handler (LLM) sees only the truncated window."""
        ext = HistoryExtension(strategy="count", max_messages=2)
        messages = _msgs(5)
        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == messages[-2:]

    @pytest.mark.asyncio
    async def test_fewer_than_limit(self):
        ext = HistoryExtension(strategy="count", max_messages=10)
        messages = _msgs(3)

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == messages

    @pytest.mark.asyncio
    async def test_preserves_system_message(self):
        ext = HistoryExtension(strategy="count", max_messages=3)
        sys_msg = FakeSystemMessage("system")
        messages = [sys_msg] + _msgs(5)

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received[0] is sys_msg
        assert len(received) == 3
        assert received[1:] == messages[-2:]

    @pytest.mark.asyncio
    async def test_system_message_only_when_budget_is_one(self):
        ext = HistoryExtension(strategy="count", max_messages=1)
        sys_msg = FakeSystemMessage("system")
        messages = [sys_msg] + _msgs(5)

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == [sys_msg]

    @pytest.mark.asyncio
    async def test_empty_list(self):
        ext = HistoryExtension(strategy="count", max_messages=5)
        result = await _run_wrap(ext, [])
        assert result["messages"][-1] == FakeMessage("response")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="max_messages must be >= 1"):
            HistoryExtension(strategy="count", max_messages=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="max_messages must be >= 1"):
            HistoryExtension(strategy="count", max_messages=-1)

    def test_requires_max_messages(self):
        with pytest.raises(ValueError, match="max_messages is required"):
            HistoryExtension(strategy="count")


# ===========================================================================
# strategy="tokens"
# ===========================================================================


class TestTokensStrategy:
    @staticmethod
    def _counter(msg: Any) -> int:
        return len(msg.content)

    @pytest.mark.asyncio
    async def test_basic_truncation(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=10, token_counter=self._counter)
        messages = [FakeMessage("aaaa"), FakeMessage("bbbb"), FakeMessage("cccc")]

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == messages[1:]

    @pytest.mark.asyncio
    async def test_preserves_system_message(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=10, token_counter=self._counter)
        sys_msg = FakeSystemMessage("system")  # 6 tokens
        messages = [sys_msg, FakeMessage("aaaa"), FakeMessage("bbbb"), FakeMessage("cccc")]

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == [sys_msg, FakeMessage("cccc")]

    @pytest.mark.asyncio
    async def test_custom_counter(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=2, token_counter=lambda _: 1)
        messages = _msgs(5)

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == messages[-2:]

    @pytest.mark.asyncio
    async def test_single_message_exceeding_budget(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=1, token_counter=self._counter)
        messages = [FakeMessage("this is a long message")]

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": messages}, handler=spy_handler, runtime=None)

        assert received == messages

    @pytest.mark.asyncio
    async def test_empty_list(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=100)
        result = await _run_wrap(ext, [])
        assert result["messages"][-1] == FakeMessage("response")

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            HistoryExtension(strategy="tokens", max_tokens=0)

    def test_requires_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens is required"):
            HistoryExtension(strategy="tokens")

    @pytest.mark.asyncio
    async def test_default_counter(self):
        ext = HistoryExtension(strategy="tokens", max_tokens=5)
        msg = FakeMessage("x" * 20)  # 5 tokens via len//4

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": [msg]}, handler=spy_handler, runtime=None)

        assert received == [msg]


# ===========================================================================
# Custom strategy via HistoryStrategy protocol
# ===========================================================================


class TestCustomStrategy:
    @pytest.mark.asyncio
    async def test_delegates_to_custom_strategy(self):
        calls: list[list[Any]] = []

        class SpyStrategy:
            def transform(self, messages: list[Any]) -> list[Any]:
                calls.append(list(messages))
                return messages[-1:]

        ext = HistoryExtension(strategy=SpyStrategy())

        received: list[Any] = []

        async def spy_handler(state: Any) -> dict[str, Any]:
            received.extend(state["messages"])
            return {"messages": []}

        await ext.wrap_model(state={"messages": ["a", "b"]}, handler=spy_handler, runtime=None)

        assert calls == [["a", "b"]]
        assert received == ["b"]

    def test_custom_class_satisfies_protocol(self):
        class MyStrategy:
            def transform(self, messages: list[Any]) -> list[Any]:
                return messages

        assert isinstance(MyStrategy(), HistoryStrategy)


# ===========================================================================
# HistoryExtension basics
# ===========================================================================


class TestHistoryExtension:
    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            HistoryExtension(strategy="summarize")  # type: ignore[arg-type]

    def test_no_tools(self):
        ext = HistoryExtension(strategy="count", max_messages=10)
        assert ext.tools == []

    def test_no_prompt(self):
        ext = HistoryExtension(strategy="count", max_messages=10)
        assert ext.prompt({}) is None

    def test_no_state_schema(self):
        ext = HistoryExtension(strategy="count", max_messages=10)
        assert ext.state_schema is None


# ===========================================================================
# HookRunner integration — wrap_model is discovered and executed
# ===========================================================================


class TestHookRunnerIntegration:
    @pytest.mark.asyncio
    async def test_wrap_model_discovered_by_hook_runner(self):
        ext = HistoryExtension(strategy="count", max_messages=2)
        runner = HookRunner([ext])
        messages = _msgs(5)

        received: list[Any] = []

        async def spy_handler(request: Any) -> dict[str, Any]:
            received.extend(request["messages"])
            return {"messages": []}

        await runner.run_wrap(
            "model", state={"messages": messages}, handler=spy_handler, runtime=None
        )

        assert received == messages[-2:]


# ===========================================================================
# ReplaceMessages — bulk replacement via add_messages REMOVE_ALL sentinel
# ===========================================================================


class TestReplaceMessages:
    @pytest.mark.asyncio
    async def test_returns_replace_messages_list(self):
        from langchain_agentkit.extensions.history.state import ReplaceMessages

        ext = HistoryExtension(strategy="count", max_messages=2)
        result = await _run_wrap(ext, _id_msgs(5))

        assert isinstance(result["messages"], ReplaceMessages)

    @pytest.mark.asyncio
    async def test_contains_remove_all_sentinel_then_kept_and_response(self):
        from langchain_core.messages import RemoveMessage

        from langchain_agentkit.extensions.history.state import ReplaceMessages

        ext = HistoryExtension(strategy="count", max_messages=2)
        msgs = _id_msgs(5)
        result = await _run_wrap(ext, msgs)

        sentinel = result["messages"]
        assert isinstance(sentinel, ReplaceMessages)
        # First element is RemoveMessage("__remove_all__")
        assert isinstance(sentinel[0], RemoveMessage)
        assert sentinel[0].id == "__remove_all__"
        # Remaining: kept window (last 2) + handler response
        assert sentinel[1:] == list(msgs[-2:]) + [FakeMessage("response")]

    @pytest.mark.asyncio
    async def test_all_fit_still_returns_sentinel(self):
        from langchain_agentkit.extensions.history.state import ReplaceMessages

        ext = HistoryExtension(strategy="count", max_messages=10)
        msgs = _id_msgs(3)
        result = await _run_wrap(ext, msgs)

        sentinel = result["messages"]
        assert isinstance(sentinel, ReplaceMessages)
        # RemoveAll + all messages + response
        assert sentinel[1:] == list(msgs) + [FakeMessage("response")]

    def test_works_with_add_messages_reducer(self):
        from langchain_core.messages import HumanMessage
        from langgraph.graph.message import add_messages

        from langchain_agentkit.extensions.history.state import ReplaceMessages

        old = [HumanMessage(content="old", id="1")]
        new = ReplaceMessages([HumanMessage(content="new", id="2")])

        result = add_messages(old, new)

        # Old message should be gone, only new remains
        assert len(result) == 1
        assert result[0].content == "new"
