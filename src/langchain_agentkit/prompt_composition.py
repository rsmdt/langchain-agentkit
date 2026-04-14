"""PromptComposition — structured return type for ``AgentKit.compose()``.

Splits the composed per-step system message into two channels:

- ``prompt`` — the system prompt contributed by the kit's base prompt and
  every extension's ``prompt()`` return, joined in declaration order.
- ``reminder`` — ephemeral guidance delivered via AgentKit's built-in
  ``<system-reminder>`` envelope (today's date plus any extension
  contributions keyed by class name).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptComposition:
    """Structured composition of the per-step system message."""

    prompt: str = ""
    reminder: str = ""
