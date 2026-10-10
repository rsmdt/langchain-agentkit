"""Tests for MemoryExtension framing and strategy delegation."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from langchain_agentkit.extensions.memory import (
    CuratedMemory,
    MemoryExtension,
    MemoryStrategy,
)


class _FakeStrategy(MemoryStrategy):
    """Records lifecycle calls and returns canned content."""

    def __init__(
        self,
        *,
        body: str | None = None,
        reminder: str | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self._body = body
        self._reminder = reminder
        self._tools = tools or []
        self.calls: list[str] = []

    @property
    def tools(self) -> list[Any]:
        return self._tools

    async def setup(self) -> None:
        self.calls.append("setup")

    async def refresh(self, *, state: dict[str, Any], runtime: Any) -> None:
        self.calls.append("refresh")

    async def after_turn(self, *, state: dict[str, Any], runtime: Any) -> None:
        self.calls.append("after_turn")

    def body(self) -> str | None:
        return self._body

    def reminder(self) -> str | None:
        return self._reminder


# --- Defaults ---


def test_default_strategy_is_curated() -> None:
    assert isinstance(MemoryExtension().strategy, CuratedMemory)


def test_prompt_none_when_strategy_empty() -> None:
    assert MemoryExtension(strategy=_FakeStrategy()).prompt({}, None) is None


def test_state_schema_is_none() -> None:
    assert MemoryExtension(strategy=_FakeStrategy()).state_schema is None


# --- Durable channel ---


def test_body_goes_to_durable_channel_with_header_and_footer() -> None:
    ext = MemoryExtension(strategy=_FakeStrategy(body="Remember: coffee first."))
    result = ext.prompt({}, None)
    assert result is not None
    assert "reminder" not in result
    durable = result["prompt"]
    assert durable.startswith("# Memory")
    assert "Remember: coffee first." in durable
    assert durable.endswith("before acting on them.")


def test_custom_header_and_footer() -> None:
    ext = MemoryExtension(
        strategy=_FakeStrategy(body="body"),
        header="## Custom Header",
        trust_footer="TRUST ME",
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert result["prompt"].startswith("## Custom Header")
    assert result["prompt"].endswith("TRUST ME")


# --- Reminder channel ---


def test_reminder_goes_to_reminder_channel() -> None:
    ext = MemoryExtension(strategy=_FakeStrategy(reminder="recalled thing"))
    result = ext.prompt({}, None)
    assert result is not None
    assert "prompt" not in result
    assert "recalled thing" in result["reminder"]


def test_reminder_only_carries_the_footer() -> None:
    # With no durable section there is nowhere else for the caveat to live.
    ext = MemoryExtension(strategy=_FakeStrategy(reminder="recalled thing"))
    result = ext.prompt({}, None)
    assert result is not None
    assert "Memory records may be stale." in result["reminder"]


def test_footer_stated_once_when_both_channels_populated() -> None:
    ext = MemoryExtension(strategy=_FakeStrategy(body="doc", reminder="recalled"))
    result = ext.prompt({}, None)
    assert result is not None
    assert "Memory records may be stale." in result["prompt"]
    assert "Memory records may be stale." not in result["reminder"]


# --- extra_sources ---


def test_extra_sources_included() -> None:
    ext = MemoryExtension(
        strategy=_FakeStrategy(body="main body"),
        extra_sources=[lambda: "extra1", lambda: "extra2"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "extra1" in result["prompt"]
    assert "extra2" in result["prompt"]


def test_extra_sources_skip_none_and_empty() -> None:
    ext = MemoryExtension(
        strategy=_FakeStrategy(body="main"),
        extra_sources=[lambda: None, lambda: "", lambda: "real"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "real" in result["prompt"]


def test_extra_sources_swallow_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> str:
        raise RuntimeError("bad source")

    ext = MemoryExtension(
        strategy=_FakeStrategy(body="main"),
        extra_sources=[boom, lambda: "ok"],
    )
    with caplog.at_level(logging.DEBUG, logger="langchain_agentkit.extensions.memory.extension"):
        result = ext.prompt({}, None)
    assert result is not None
    assert "ok" in result["prompt"]


def test_extra_sources_alone_produce_durable_section() -> None:
    ext = MemoryExtension(
        strategy=_FakeStrategy(),
        extra_sources=[lambda: "only this"],
    )
    result = ext.prompt({}, None)
    assert result is not None
    assert "only this" in result["prompt"]


# --- Delegation ---


def test_tools_delegate_to_strategy() -> None:
    sentinel = object()
    ext = MemoryExtension(strategy=_FakeStrategy(tools=[sentinel]))
    assert ext.tools == [sentinel]


@pytest.mark.asyncio
async def test_lifecycle_hooks_delegate_to_strategy() -> None:
    strategy = _FakeStrategy()
    ext = MemoryExtension(strategy=strategy)

    await ext.setup()
    await ext.before_model(state={}, runtime=None)
    await ext.after_run(state={}, runtime=None)

    assert strategy.calls == ["setup", "refresh", "after_turn"]


@pytest.mark.asyncio
async def test_base_strategy_lifecycle_is_a_noop() -> None:
    # A strategy that overrides nothing must still satisfy every call site.
    ext = MemoryExtension(strategy=MemoryStrategy())
    await ext.setup()
    await ext.before_model(state={}, runtime=None)
    await ext.after_run(state={}, runtime=None)
    assert ext.prompt({}, None) is None
    assert ext.tools == []
