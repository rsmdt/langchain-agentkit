"""LLM-input filter helper for subagent trace messages.

The :func:`trace_hidden_strategy` persists every subagent AIMessage into
parent state tagged with ``{prefix}_hidden_from_llm=True``. The complementary
read-side filter lives on :class:`AgentsExtension` itself as a
``wrap_model`` hook — see ``AgentsExtension.wrap_model``. This module
exposes the underlying list-transformation primitive as a public helper
for advanced users who implement custom output strategies and want to
reuse the filter logic directly (e.g. in a custom graph handler that
bypasses the standard ``AgentsExtension`` onion).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
