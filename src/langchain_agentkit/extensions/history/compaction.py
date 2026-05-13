"""CompactionStrategy — collapse history into one summary message when context fills.

When the conversation's estimated token count exceeds
``context_window - reserve_tokens``, this strategy returns
``[<system_message?>, HumanMessage(<summary>)]`` — i.e. it replaces the
entire history with a single synthetic summary so the next LLM call
runs against a fresh context window. A leading ``SystemMessage`` is
preserved verbatim (it's the agent's persona, not part of history).

Summaries chain across compaction rounds via ``previous_summary``
plumbing inside :func:`_summarizer.generate_summary` so the structured
checkpoint accumulates rather than degrading round after round.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from langchain_agentkit.extensions.history._file_ops import (
    compute_file_lists,
    extract_file_ops,
    format_file_operations,
)
from langchain_agentkit.extensions.history._summarizer import generate_summary
from langchain_agentkit.extensions.history._token_accounting import (
    estimate_context_tokens,
    should_compact,
)
from langchain_agentkit.extensions.history.strategies import _is_system_message

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel

_logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW: int = 128_000
DEFAULT_RESERVE_TOKENS: int = 16_384


class CompactionStrategy:
    """Summarize the entire conversation into one ``HumanMessage`` when context fills.

    Args:
        context_window: Model's context window in tokens. Used as the
            static fallback when ``context_window_resolver`` is not set
            or raises.
        context_window_resolver: Optional callable returning the model's
            current context window in tokens — useful when the value
            comes from a provider library (e.g. LiteLLM). Resolved once
            on first use and cached.
        reserve_tokens: Compaction fires when the estimated context
            usage exceeds ``context_window - reserve_tokens``.
        custom_instructions: Optional extra guidance appended to the
            summarization prompt — e.g. "Focus on file paths edited
            and unresolved errors."
        summarizer_llm: Optional ``BaseChatModel`` for summarization —
            typically a cheaper model than the main agent. When ``None``,
            the kit's main LLM is used (wired via :meth:`setup`).
    """

    def __init__(
        self,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        context_window_resolver: Callable[[], int] | None = None,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        custom_instructions: str | None = None,
        summarizer_llm: BaseChatModel | None = None,
    ) -> None:
        self._context_window = context_window
        self._context_window_resolver = context_window_resolver
        self._reserve_tokens = reserve_tokens
        self._custom_instructions = custom_instructions
        self._summarizer_llm = summarizer_llm
        self._llm_getter: Callable[[], BaseChatModel] | None = None
        self._last_summary: str | None = None
        self._resolved_context_window: int | None = None
        self._lock = asyncio.Lock()

    async def setup(
        self,
        *,
        llm_getter: Callable[[], BaseChatModel] | None = None,
    ) -> None:
        """Capture the kit's LLM getter as a summarizer fallback."""
        if llm_getter is not None:
            self._llm_getter = llm_getter

    def contribute_prompt(self) -> dict[str, str]:
        """System-prompt reminder describing compaction behavior to the LLM."""
        return {
            "reminder": (
                "When the conversation grows large, all prior messages are "
                "summarized into a single synthetic message. Write exact "
                "file paths, function names, and error messages into your "
                "reply text — they survive compaction; raw tool output may not."
            )
        }

    async def transform(self, messages: list[Any], *, runtime: Any) -> list[Any]:
        if len(messages) < 2:
            return messages

        ctx_window = self._resolve_context_window()
        usage = estimate_context_tokens(messages)
        if not should_compact(usage.tokens, ctx_window, self._reserve_tokens):
            return messages

        if _is_system_message(messages[0]):
            system_head = [messages[0]]
            body = messages[1:]
        else:
            system_head = []
            body = messages

        if not body:
            return messages

        async with self._lock:
            llm = self._resolve_summarizer_llm()
            summary_body = await generate_summary(
                body,
                llm,
                previous_summary=self._last_summary,
                custom_instructions=self._custom_instructions,
            )

            ops = extract_file_ops(body)
            read_files, modified_files = compute_file_lists(ops)
            summary_body += format_file_operations(read_files, modified_files)

            self._last_summary = summary_body
            envelope = self._render_envelope(summary_body)
            return [*system_head, HumanMessage(content=envelope)]

    def _resolve_context_window(self) -> int:
        if self._resolved_context_window is not None:
            return self._resolved_context_window
        if self._context_window_resolver is not None:
            try:
                self._resolved_context_window = int(self._context_window_resolver())
            except Exception:  # noqa: BLE001 — resolver failure is non-fatal
                _logger.warning("context_window_resolver raised; using static value")
                self._resolved_context_window = self._context_window
        else:
            self._resolved_context_window = self._context_window
        return self._resolved_context_window

    def _resolve_summarizer_llm(self) -> Any:
        if self._summarizer_llm is not None:
            return self._summarizer_llm
        if self._llm_getter is None:
            raise RuntimeError(
                "CompactionStrategy requires either summarizer_llm= at "
                "construction or a kit-level LLM (wired via setup())."
            )
        return self._llm_getter()

    @staticmethod
    def _render_envelope(body: str) -> str:
        return (
            "<compaction-summary>\n"
            "The earlier conversation was summarized to conserve context. "
            "Treat the following as authoritative history.\n\n"
            f"{body}\n"
            "</compaction-summary>"
        )
