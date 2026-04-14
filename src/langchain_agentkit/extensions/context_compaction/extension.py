"""ContextCompactionExtension — evict old tool results from the LLM request.

Keeps the content of the N most recent ``ToolMessage`` entries intact and
replaces the content of older ones with a short placeholder before handing
the message list to the model. Message envelopes (``tool_call_id``,
``name``, ``id``) are preserved so LangGraph reducers and AIMessage
linkage stay valid.

Eviction is applied only to the per-turn LLM request via ``wrap_model`` —
graph state is not mutated. Each turn re-runs eviction on the latest
state, so the policy is a pure function of the current message list and
requires no cross-turn bookkeeping.

Pairs with the "Tool results" guidance in ``CoreBehaviorExtension``,
which instructs the model to write important findings into its own reply
text before the underlying tool output can be evicted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt import ToolRuntime

EVICTED_MARKER = "[tool result cleared to conserve context]"


class ContextCompactionExtension(Extension):
    """Evict old tool results from the LLM context window.

    Args:
        keep_recent: Number of most recent ``ToolMessage`` entries to
            retain with full content. Must be >= 1. Defaults to 5.
    """

    def __init__(self, keep_recent: int = 5) -> None:
        if keep_recent < 1:
            raise ValueError(f"keep_recent must be >= 1, got {keep_recent}")
        self._keep_recent = keep_recent

    @property
    def keep_recent(self) -> int:
        return self._keep_recent

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> dict[str, str]:
        return {
            "reminder": (
                "Old tool results are automatically cleared from context to "
                f"conserve tokens. The {self._keep_recent} most recent tool "
                "results are kept with full content; older ones are replaced "
                "with a placeholder. Write important findings into your "
                "reply text so they survive eviction."
            )
        }

    async def wrap_model(
        self,
        *,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        messages = state.get("messages") or []
        redacted = redact_old_tool_messages(messages, self._keep_recent)
        if redacted is messages:
            return await handler(state)
        return await handler({**state, "messages": redacted})


def redact_old_tool_messages(
    messages: list[Any],
    keep_recent: int,
) -> list[Any]:
    """Return a new list with old ``ToolMessage`` contents replaced.

    The ``keep_recent`` most recent ``ToolMessage`` entries retain their
    content. Older ones are copied with ``content`` replaced by
    :data:`EVICTED_MARKER` — envelope fields (``tool_call_id``, ``name``,
    ``id``) are preserved. Non-``ToolMessage`` entries are never touched.

    Already-redacted messages are skipped to keep the operation idempotent.
    Returns the original list unchanged when no redaction is needed, so
    callers can cheaply detect a no-op with an ``is`` check.
    """
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    if len(tool_indices) <= keep_recent:
        return messages
    evict_before = tool_indices[-keep_recent]
    evict_set = {i for i in tool_indices if i < evict_before}
    if not evict_set:
        return messages
    new_messages = list(messages)
    changed = False
    for i in evict_set:
        msg = new_messages[i]
        if msg.content == EVICTED_MARKER:
            continue
        new_messages[i] = msg.model_copy(update={"content": EVICTED_MARKER})
        changed = True
    return new_messages if changed else messages
