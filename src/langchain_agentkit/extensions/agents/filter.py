"""LLM-input filter for subagent trace messages.

The :func:`trace_hidden_strategy` persists every subagent AIMessage into
parent state tagged with ``{prefix}_hidden_from_llm=True``. This module
provides the complementary read-side filter — stripping those messages
from the per-request message list before the parent LLM is invoked, so
the model's context window contains only the terminal ``ToolMessage``.

Two primitives:

* :func:`strip_hidden_from_llm` — a pure function over a message list.
  Use it directly from a handler or a custom ``wrap_model`` hook.

* :class:`HideSubagentTraceExtension` — a standalone extension that
  applies the filter as its ``wrap_model`` hook. Consumers who use
  :class:`AgentsExtension` with the default ``trace_hidden`` output
  mode should install this *inner* to any history/compaction
  extensions so persistence (via ``ReplaceMessages``) still sees the
  full trace.

Ordering note
-------------
In AgentKit the declaration order of an extensions list maps to the
outer-to-inner order of the ``wrap_model`` onion. For the trace-hidden
design to work:

    [..., HistoryExtension, ContextCompactionExtension, HideSubagentTraceExtension]

History's ``kept`` (which becomes the replaced state) is computed from
the un-filtered state and therefore retains every subagent message for
persistence. The filter then strips those messages from what the LLM
actually sees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import BaseMessage


DEFAULT_METADATA_PREFIX = "agentkit"


def strip_hidden_from_llm(
    messages: list[BaseMessage] | tuple[BaseMessage, ...],
    *,
    metadata_prefix: str = DEFAULT_METADATA_PREFIX,
) -> list[BaseMessage]:
    """Return a new list with all hidden-tagged messages removed.

    A message is considered hidden when its ``response_metadata`` contains
    ``{metadata_prefix}_hidden_from_llm=True``. Other messages pass
    through unchanged.

    The filter never mutates the inputs. Safe to call on every turn.
    """
    flag = f"{metadata_prefix}_hidden_from_llm"
    out: list[BaseMessage] = []
    for msg in messages:
        meta = getattr(msg, "response_metadata", None) or {}
        if meta.get(flag) is True:
            continue
        out.append(msg)
    return out


class HideSubagentTraceExtension(Extension):
    """Extension that hides tagged subagent messages from the parent LLM.

    Pair with :class:`AgentsExtension` configured for ``trace_hidden``
    output. Place *inner* to history/compaction extensions so the
    persisted state keeps the full trace while the LLM-facing view
    strips it.

    Args:
        metadata_prefix: Prefix used by AgentsExtension for tag keys.
            Must match the prefix the subagent strategy uses. Defaults
            to ``"agentkit"``.
    """

    def __init__(self, *, metadata_prefix: str = DEFAULT_METADATA_PREFIX) -> None:
        self._metadata_prefix = metadata_prefix

    @property
    def metadata_prefix(self) -> str:
        return self._metadata_prefix

    async def wrap_model(
        self,
        *,
        state: Any,
        handler: Callable[[Any], Awaitable[Any]],
        runtime: Any,
    ) -> Any:
        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return await handler(state)

        filtered = strip_hidden_from_llm(messages, metadata_prefix=self._metadata_prefix)
        if len(filtered) == len(messages):
            return await handler(state)

        return await handler({**state, "messages": filtered})
