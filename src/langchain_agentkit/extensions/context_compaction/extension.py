"""ContextCompactionExtension — summarize the old prefix to reclaim tokens.

When the conversation approaches the model's context window, we bisect
the message list at a safe user/assistant boundary, hand the to-be-dropped
prefix to the LLM with a structured summarization prompt, and replace it
with a single synthetic ``HumanMessage`` carrying the summary plus a
file-operation index.

Key invariants
--------------

* **Pure over state.** Compaction runs inside ``wrap_model`` and mutates
  only the per-step LLM request. Graph state is never rewritten, so
  checkpoints remain faithful to the real history.
* **Content-hash cache.** Summarization is expensive. The summary for a
  given prefix is cached on the extension instance keyed by a stable hash
  of the serialized prefix; subsequent turns reuse it until new messages
  shift the cut point. The cache is scope-bound to the extension
  instance — share an instance within a conversation, create a new one
  for a new conversation.
* **Token budget from metadata.** The extension consumes
  :class:`ModelMetadata` via :func:`setup` or a caller-provided
  ``context_window_resolver`` callback. The fallback is
  ``DEFAULT_CONTEXT_WINDOW`` when neither is available.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.context_compaction.cutpoint import (
    CutPoint,
    find_cut_point,
)
from langchain_agentkit.extensions.context_compaction.file_ops import (
    compute_file_lists,
    extract_file_ops,
    format_file_operations,
)
from langchain_agentkit.extensions.context_compaction.summarizer import (
    generate_summary,
    generate_turn_prefix_summary,
    serialize_conversation,
)
from langchain_agentkit.extensions.context_compaction.token_accounting import (
    estimate_context_tokens,
    should_compact,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt import ToolRuntime

_logger = logging.getLogger(__name__)

# Conservative default when no ModelMetadata is resolvable — a cheap way
# to avoid over-triggering on unknown models.
DEFAULT_CONTEXT_WINDOW: int = 128_000
DEFAULT_RESERVE_TOKENS: int = 16_384
DEFAULT_KEEP_RECENT_TOKENS: int = 20_000


@dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Trigger + budget knobs for :class:`ContextCompactionExtension`."""

    enabled: bool = True
    reserve_tokens: int = DEFAULT_RESERVE_TOKENS
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS
    custom_instructions: str | None = None


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


class ContextCompactionExtension(Extension):
    """Summarize old history when the context window fills up.

    Args:
        settings: Trigger / budget configuration. Defaults to
            :data:`DEFAULT_COMPACTION_SETTINGS`.
        context_window_resolver: Optional callable returning the context
            window (in tokens). Takes precedence over
            :class:`ModelMetadata`. Use when your model isn't in
            :data:`langchain_agentkit.core.DEFAULT_REGISTRY` and you
            don't want to :func:`register_model` globally.
        summarizer_llm: Optional ``BaseChatModel`` to use for
            summarization — typically a cheaper model than the main
            agent. Defaults to the kit's main LLM.
    """

    def __init__(
        self,
        *,
        settings: CompactionSettings = DEFAULT_COMPACTION_SETTINGS,
        context_window_resolver: Callable[[], int] | None = None,
        summarizer_llm: Any | None = None,
    ) -> None:
        self._settings = settings
        self._context_window_resolver = context_window_resolver
        self._summarizer_llm = summarizer_llm
        self._cache: dict[str, str] = {}
        self._last_summary: str | None = None
        self._resolved_context_window: int | None = None
        self._llm_getter: Callable[[], Any] | None = None
        self._in_flight: asyncio.Lock = asyncio.Lock()

    async def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Capture the kit's LLM getter and model metadata during assembly."""
        llm_getter = kwargs.get("llm_getter")
        if callable(llm_getter):
            self._llm_getter = llm_getter
        metadata_getter = kwargs.get("model_metadata_getter")
        if self._context_window_resolver is None and callable(metadata_getter):
            # Capture lazily so summarizer LLM resolution doesn't force
            # early model instantiation when the kit only has a string.
            def _resolver() -> int:
                meta = metadata_getter()
                return int(meta.context_window) if meta is not None else DEFAULT_CONTEXT_WINDOW

            self._context_window_resolver = _resolver

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> dict[str, str]:
        return {
            "reminder": (
                "Old conversation turns are automatically summarized into a "
                "synthetic checkpoint when context fills up. Write exact file "
                "paths, function names, and error messages into your reply "
                "text — they survive compaction; raw tool output may not."
            )
        }

    async def wrap_model(
        self,
        *,
        state: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        if not self._settings.enabled:
            return await handler(state)
        messages = list(state.get("messages") or [])
        if len(messages) < 2:
            return await handler(state)

        ctx_window = self._context_window()
        usage = estimate_context_tokens(messages)
        if not should_compact(usage.tokens, ctx_window, self._settings.reserve_tokens):
            return await handler(state)

        cut = find_cut_point(messages, 0, len(messages), self._settings.keep_recent_tokens)
        if cut.first_kept_index <= 0:
            return await handler(state)

        summary_message = await self._build_or_reuse_summary(messages, cut)
        if summary_message is None:
            return await handler(state)

        kept = messages[cut.first_kept_index :]
        compacted_messages = [summary_message, *kept]
        return await handler({**state, "messages": compacted_messages})

    # --- Internal helpers ---------------------------------------------------

    def _context_window(self) -> int:
        if self._resolved_context_window is not None:
            return self._resolved_context_window
        if self._context_window_resolver is not None:
            try:
                self._resolved_context_window = int(self._context_window_resolver())
            except Exception:  # noqa: BLE001 — resolver failure is non-fatal
                _logger.warning("context_window_resolver raised; using default")
                self._resolved_context_window = DEFAULT_CONTEXT_WINDOW
        else:
            self._resolved_context_window = DEFAULT_CONTEXT_WINDOW
        return self._resolved_context_window

    def _resolve_summarizer_llm(self) -> Any:
        if self._summarizer_llm is not None:
            return self._summarizer_llm
        if self._llm_getter is None:
            raise RuntimeError(
                "ContextCompactionExtension requires either summarizer_llm= "
                "at construction or a kit-level LLM wired via setup()."
            )
        return self._llm_getter()

    def _prefix_hash(self, prefix: list[Any]) -> str:
        payload = serialize_conversation(prefix)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _build_or_reuse_summary(
        self,
        messages: list[Any],
        cut: CutPoint,
    ) -> HumanMessage | None:
        prefix = messages[: cut.first_kept_index]
        if not prefix:
            return None
        key = self._prefix_hash(prefix)
        if key in self._cache:
            return HumanMessage(content=self._cache[key])

        # Lock so concurrent model calls don't duplicate summarization work
        # for the same prefix.
        async with self._in_flight:
            if key in self._cache:  # woke up after another caller populated it
                return HumanMessage(content=self._cache[key])

            llm = self._resolve_summarizer_llm()
            history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_index
            main_slice = messages[:history_end]

            # Skip the main summary call when the slice is empty — e.g. a
            # split turn where the dropped prefix is the opening turn.
            main_summary = (
                await generate_summary(
                    main_slice,
                    llm,
                    reserve_tokens=self._settings.reserve_tokens,
                    previous_summary=self._last_summary,
                    custom_instructions=self._settings.custom_instructions,
                )
                if main_slice
                else ""
            )

            if cut.is_split_turn and cut.turn_start_index >= 0:
                turn_prefix = messages[cut.turn_start_index : cut.first_kept_index]
                turn_summary = await generate_turn_prefix_summary(turn_prefix, llm)
                if main_summary:
                    summary_body = (
                        f"{main_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_summary}"
                    )
                else:
                    summary_body = f"**Turn Context (split turn):**\n\n{turn_summary}"
            else:
                summary_body = main_summary

            # File-operation index spans everything we're dropping — both
            # the main prefix and the turn prefix when split.
            drop_end = cut.first_kept_index
            ops = extract_file_ops(messages[:drop_end])
            read_files, modified_files = compute_file_lists(ops)
            summary_body += format_file_operations(read_files, modified_files)

            rendered = self._render_summary_envelope(summary_body)
            self._cache[key] = rendered
            if main_summary:
                self._last_summary = main_summary
            return HumanMessage(content=rendered)

    @staticmethod
    def _render_summary_envelope(summary_body: str) -> str:
        return (
            "<compaction-summary>\n"
            "The earlier conversation was summarized to conserve context. "
            "Treat the following as authoritative history.\n\n"
            f"{summary_body}\n"
            "</compaction-summary>"
        )

    @property
    def cache_size(self) -> int:
        """Number of cached summaries (exposed for tests and diagnostics)."""
        return len(self._cache)

    def invalidate_cache(self) -> None:
        """Drop every cached summary — call when starting a new session."""
        self._cache.clear()
        self._last_summary = None
