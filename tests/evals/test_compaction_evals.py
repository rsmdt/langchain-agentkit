"""LLM behavioral evals for CoreBehavior "Tool results" guidance and
``ContextCompactionExtension`` redaction.

Two behaviors are evaluated:

1. **Summarize**: with CoreBehavior's "Tool results" guidance present, the
   model extracts concrete facts from a tool result into its own reply
   text rather than referencing the result implicitly.
2. **Compaction**: with ``ContextCompactionExtension`` wrapping the model
   call, old ``ToolMessage`` content is invisible to the LLM — the model
   cannot quote facts that exist only in evicted results.

Run::

    pytest tests/evals/test_compaction_evals.py -v -m eval

Skipped cleanly when ``OPENAI_API_KEY`` is unset.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

pytestmark = pytest.mark.eval

try:
    from langchain_openai import ChatOpenAI

    _HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False

SKIP_REASON = "Requires OPENAI_API_KEY and langchain-openai"
MODEL_NAME = "gpt-4o-mini"


def _llm() -> Any:
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


def _make_kit(extensions: list[Any]) -> Any:
    from langchain_agentkit import AgentKit
    from langchain_agentkit.agent_kit import run_extension_setup

    kit = AgentKit(extensions=extensions)
    asyncio.run(run_extension_setup(kit))
    return kit


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _best_of_three(fn: Any) -> tuple[int, list[str]]:
    passes = 0
    comments: list[str] = []
    for i in range(3):
        try:
            ok, comment = fn()
        except Exception as exc:  # pragma: no cover
            ok, comment = False, f"trial {i}: raised {exc!r}"
        if ok:
            passes += 1
        else:
            comments.append(f"trial {i}: {comment}")
    return passes, comments


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    parts: list[str] = []
    for part in msg.content:
        if isinstance(part, dict) and "text" in part:
            parts.append(str(part["text"]))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(parts)


async def _invoke_with_history(kit: Any, history: list[Any]) -> AIMessage:
    """Invoke LLM with an injected message history, applying kit.compose()."""
    state: dict[str, Any] = {"messages": history}
    composition = kit.compose(state)
    messages: list[Any] = []
    if composition.prompt:
        messages.append(SystemMessage(content=composition.prompt))
    messages.extend(history)
    if composition.reminder:
        messages.append(HumanMessage(content=composition.reminder))
    response = await _llm().ainvoke(messages)
    assert isinstance(response, AIMessage)
    return response


async def _invoke_through_wrap_model(
    kit: Any,
    compaction_ext: Any,
    history: list[Any],
) -> AIMessage:
    """Invoke via ``compaction_ext.wrap_model`` — the real eviction path."""
    state: dict[str, Any] = {"messages": history}
    composition = kit.compose(state)

    async def handler(s: dict[str, Any]) -> AIMessage:
        messages: list[Any] = []
        if composition.prompt:
            messages.append(SystemMessage(content=composition.prompt))
        messages.extend(s["messages"])
        if composition.reminder:
            messages.append(HumanMessage(content=composition.reminder))
        response = await _llm().ainvoke(messages)
        assert isinstance(response, AIMessage)
        return response

    return await compaction_ext.wrap_model(state=state, handler=handler, runtime=None)


# ---------------------------------------------------------------------------
# Summarize eval — CoreBehavior "Tool results" guidance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestSummarizeToolResultsGuidance:
    """With CoreBehavior's guidance, the model quotes concrete facts from
    tool output in its reply text."""

    def test_model_extracts_file_and_line_into_reply(self) -> None:
        from langchain_agentkit.extensions.core_behavior import (
            CoreBehaviorExtension,
        )

        kit = _make_kit([CoreBehaviorExtension()])

        tool_output = (
            "src/auth.py:58: if payload['exp'] < str(time.time()):\n"
            "src/auth.py:59:     raise TokenExpired()\n"
            "src/auth.py:42: def verify_token(token: str) -> Claims:\n"
        )
        history: list[Any] = [
            HumanMessage(content="Find the bug in verify_token and explain it."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "grep",
                        "args": {"pattern": "verify_token|exp"},
                    }
                ],
            ),
            ToolMessage(content=tool_output, tool_call_id="call_1", name="grep"),
            HumanMessage(
                content=(
                    "Based on what you just found, state the bug plainly: what's wrong, and where?"
                )
            ),
        ]

        def trial() -> tuple[bool, str]:
            msg = _run(_invoke_with_history(kit, history))
            reply = _text(msg).lower()
            has_line = "58" in reply
            has_location = "auth.py" in reply or "verify_token" in reply
            has_expr = "str(time.time())" in reply or "payload['exp']" in reply
            if not (has_line and has_location and has_expr):
                return False, (
                    f"reply missing tool-result grounding: line={has_line}, "
                    f"location={has_location}, expr={has_expr}; "
                    f"reply={reply!r}"
                )
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# Compaction eval — old ToolMessage content is invisible post-redaction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENAI, reason=SKIP_REASON)
class TestContextCompactionHidesOldToolResults:
    """With keep_recent=1, content of older ToolMessages must be
    unrecoverable by the LLM on the current turn."""

    def _history(self) -> list[Any]:
        return [
            HumanMessage(content="Gather three codes for me."),
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "fetch", "args": {"slot": 1}}],
            ),
            ToolMessage(
                content="OLDEST_SECRET=ALPHA-9182",
                tool_call_id="c1",
                name="fetch",
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "c2", "name": "fetch", "args": {"slot": 2}}],
            ),
            ToolMessage(
                content="MIDDLE_SECRET=BRAVO-5540",
                tool_call_id="c2",
                name="fetch",
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "c3", "name": "fetch", "args": {"slot": 3}}],
            ),
            ToolMessage(
                content="NEWEST_SECRET=CHARLIE-7731",
                tool_call_id="c3",
                name="fetch",
            ),
            HumanMessage(
                content=(
                    "Report every secret you received, each on its own line, "
                    "in the exact form KEY=VALUE."
                )
            ),
        ]

    def test_evicted_secrets_are_not_quoted(self) -> None:
        from langchain_agentkit.extensions.context_compaction import (
            ContextCompactionExtension,
        )
        from langchain_agentkit.extensions.core_behavior import (
            CoreBehaviorExtension,
        )

        compaction = ContextCompactionExtension(keep_recent=1)
        kit = _make_kit([CoreBehaviorExtension(), compaction])

        def trial() -> tuple[bool, str]:
            msg = _run(_invoke_through_wrap_model(kit, compaction, self._history()))
            reply = _text(msg)
            leaks = [s for s in ("ALPHA-9182", "BRAVO-5540") if s in reply]
            if leaks:
                return False, f"evicted secrets leaked: {leaks}; reply={reply!r}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"

    def test_most_recent_secret_is_still_quoted(self) -> None:
        from langchain_agentkit.extensions.context_compaction import (
            ContextCompactionExtension,
        )
        from langchain_agentkit.extensions.core_behavior import (
            CoreBehaviorExtension,
        )

        compaction = ContextCompactionExtension(keep_recent=1)
        kit = _make_kit([CoreBehaviorExtension(), compaction])

        def trial() -> tuple[bool, str]:
            msg = _run(_invoke_through_wrap_model(kit, compaction, self._history()))
            reply = _text(msg)
            if "CHARLIE-7731" not in reply:
                return False, f"most-recent secret absent from reply={reply!r}"
            return True, ""

        passes, comments = _best_of_three(trial)
        assert passes >= 2, f"{passes}/3 passed; {comments}"


# ---------------------------------------------------------------------------
# Compaction unit-level eval — no API call, verifies the marker replaces
# content on the serialized LLM input path.
# ---------------------------------------------------------------------------


class TestCompactionProducesEvictedMarkerOnLLMInput:
    """Deterministic check: after wrap_model, the serialized message list
    the LLM would see contains the marker in place of old tool content."""

    def test_marker_visible_in_handler_state(self) -> None:
        from langchain_agentkit.extensions.context_compaction import (
            ContextCompactionExtension,
        )
        from langchain_agentkit.extensions.context_compaction.extension import (
            EVICTED_MARKER,
        )

        ext = ContextCompactionExtension(keep_recent=1)
        captured: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            captured["messages"] = state["messages"]
            return {}

        history: list[Any] = [
            ToolMessage(content="OLD-A", tool_call_id="c1", name="t"),
            ToolMessage(content="OLD-B", tool_call_id="c2", name="t"),
            ToolMessage(content="RECENT", tool_call_id="c3", name="t"),
        ]
        asyncio.run(ext.wrap_model(state={"messages": history}, handler=handler, runtime=None))

        contents = [m.content for m in captured["messages"]]
        assert contents == [EVICTED_MARKER, EVICTED_MARKER, "RECENT"]
