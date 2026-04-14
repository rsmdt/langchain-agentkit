"""PromptComposition — structured return type for ``AgentKit.compose()``.

Splits the composed system prompt into three channels:

- ``static`` — content that rarely changes between turns (base prompt,
  tool catalogs, stable guidance). Safe to place at the head of the
  system message for maximum prompt-cache reuse.
- ``dynamic`` — content that renders live state (task list, team
  status). Placed after the static block so it does not invalidate
  the cache of the preceding content.
- ``reminder`` — ephemeral guidance injected by ``AgentKit``'s built-in
  reminder channel. Populated by a later unit; always empty for now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptComposition:
    """Structured composition of the per-step system prompt.

    All three fields default to empty string. Use :attr:`joined` to get
    the combined ``static`` + ``dynamic`` prompt text with empty parts
    filtered.
    """

    static: str = ""
    dynamic: str = ""
    reminder: str = ""

    @property
    def joined(self) -> str:
        """Return ``static`` and ``dynamic`` joined with double newline.

        Empty parts are filtered so the result never contains leading,
        trailing, or duplicated blank paragraphs.
        """
        parts = [p for p in (self.static, self.dynamic) if p]
        return "\n\n".join(parts)
