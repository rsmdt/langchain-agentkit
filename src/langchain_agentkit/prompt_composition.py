"""PromptComposition — structured return type for ``AgentKit.compose()``.

A single channel: the system prompt assembled from the kit's base prompt,
every extension's ``prompt()`` return, and a kit-level date contribution.
Re-rendered per step so that dynamic state (task lists, compaction status,
skill rosters) reflects the current moment.

There is no separate reminder / ephemeral channel. Extensions that want
per-turn content put it in the system prompt directly; ``kit.compose()``
is called once per LLM call anyway, so the cost is the same.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PromptComposition"]


@dataclass(frozen=True)
class PromptComposition:
    """Composed per-step system prompt."""

    prompt: str = ""
