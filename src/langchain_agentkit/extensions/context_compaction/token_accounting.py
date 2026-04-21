"""Token accounting — estimate conversation size for compaction triggers.

We prefer real provider-reported usage when it's available on the last
assistant message (LangChain surfaces it at ``AIMessage.usage_metadata``
for providers that report it). Trailing messages after that point get
estimated via a conservative chars / 4 heuristic. If no usage is present
anywhere in the history, we estimate the full transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

# Rough per-image token cost — images accounted as roughly 1200 tokens.
_IMAGE_CHAR_EQUIVALENT = 4800


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    """Result of :func:`estimate_context_tokens`."""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def _block_text(block: Any) -> str:
    """Extract text from a LangChain content block regardless of shape."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        btype = block.get("type")
        if btype in ("text", "reasoning"):
            return str(block.get("text") or block.get("reasoning") or "")
        if btype == "tool_use":
            name = str(block.get("name", ""))
            args = block.get("input") or block.get("arguments") or {}
            try:
                import json

                return f"{name}{json.dumps(args, default=str)}"
            except (TypeError, ValueError):
                return name
        if btype == "image" or btype == "image_url":
            return "\0" * _IMAGE_CHAR_EQUIVALENT
    return ""


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(_block_text(b)) for b in content)
    return 0


def estimate_tokens(message: Any) -> int:
    """Approximate token count for ``message`` using chars / 4.

    Handles LangChain's polymorphic content (str or list-of-dicts) and
    attaches a fixed image surcharge when an image block is present.
    Tool-call payloads on ``AIMessage`` are counted toward the message
    size so compaction triggers see tool-heavy turns realistically.
    """
    chars = _content_chars(getattr(message, "content", ""))
    if isinstance(message, AIMessage):
        for call in getattr(message, "tool_calls", None) or []:
            chars += len(call.get("name", ""))
            try:
                import json

                chars += len(json.dumps(call.get("args", {}), default=str))
            except (TypeError, ValueError):
                pass
        for call in getattr(message, "invalid_tool_calls", None) or []:
            chars += len(call.get("name", "")) + len(str(call.get("args", "")))
    return ceil(chars / 4)


def _extract_usage_total(message: Any) -> int | None:
    """Return the total-tokens count from an AIMessage, or ``None``."""
    if not isinstance(message, AIMessage):
        return None
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total
        ins = usage.get("input_tokens") or 0
        out = usage.get("output_tokens") or 0
        if ins or out:
            return int(ins) + int(out)
    meta = getattr(message, "response_metadata", None) or {}
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            total = token_usage.get("total_tokens") or token_usage.get("totalTokens")
            if isinstance(total, int):
                return total
    return None


def _last_usage_info(messages: Sequence[Any]) -> tuple[int, int] | None:
    for idx in range(len(messages) - 1, -1, -1):
        total = _extract_usage_total(messages[idx])
        if total is not None:
            return total, idx
    return None


def estimate_context_tokens(messages: Sequence[Any]) -> ContextUsageEstimate:
    """Best-effort token count for the conversation.

    Prefers real usage from the latest assistant message and estimates
    only the messages added after it.
    """
    info = _last_usage_info(messages)
    if info is None:
        total = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(
            tokens=total,
            usage_tokens=0,
            trailing_tokens=total,
            last_usage_index=None,
        )
    usage_total, idx = info
    trailing = sum(estimate_tokens(m) for m in messages[idx + 1 :])
    return ContextUsageEstimate(
        tokens=usage_total + trailing,
        usage_tokens=usage_total,
        trailing_tokens=trailing,
        last_usage_index=idx,
    )


def should_compact(ctx_tokens: int, context_window: int, reserve_tokens: int) -> bool:
    """Trigger when remaining headroom falls below ``reserve_tokens``."""
    return ctx_tokens > context_window - reserve_tokens


# Message type helpers used by cutpoint / serialization logic below.
_USER_LIKE = (HumanMessage, SystemMessage)
_ASSISTANT_LIKE = (AIMessage,)
_TOOL_LIKE = (ToolMessage,)


def is_user_like(msg: Any) -> bool:
    return isinstance(msg, _USER_LIKE)


def is_assistant_like(msg: Any) -> bool:
    return isinstance(msg, _ASSISTANT_LIKE)


def is_tool_like(msg: Any) -> bool:
    return isinstance(msg, _TOOL_LIKE)
