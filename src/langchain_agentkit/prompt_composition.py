"""PromptComposition — structured return type for ``AgentKit.compose()``.

Two channels, both re-rendered per step so dynamic state (task lists,
compaction status, skill rosters, turn budgets) reflects the current moment:

- ``prompt`` — the durable system prompt assembled from the kit's base
  prompt and every extension's ``prompt()`` ``str`` / ``dict["prompt"]``
  return, joined in declaration order. Delivered as the system message.
- ``reminder`` — per-turn ephemeral guidance assembled from every
  extension's ``dict["reminder"]`` return, wrapped in a ``<reminder>``
  envelope. Delivered by appending it to the *last* message of the
  conversation each step (separated by ``---``), never persisted to state.
  Empty string when no extension contributes a reminder.

The split exists because the two have different lifetimes and positions:
the prompt is stable system-channel guidance, while the reminder is
volatile per-turn state that benefits from riding at the tail of the
conversation (strongest recency attention) without polluting persisted
history or an extension's truncation window.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PromptComposition"]


@dataclass(frozen=True)
class PromptComposition:
    """Composed per-step prompt: durable system prompt + ephemeral reminder."""

    prompt: str = ""
    reminder: str = ""
