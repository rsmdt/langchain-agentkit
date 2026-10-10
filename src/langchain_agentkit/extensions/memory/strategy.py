"""Strategy seam for :class:`MemoryExtension` — where memory comes from.

The extension owns *framing*: the header, the staleness caveat, and any
caller-supplied extra sources.  A strategy owns *sourcing*: where memory
lives, how it is read, and whether the agent can write it.

Two strategies ship:

- :class:`~langchain_agentkit.extensions.memory.curated.CuratedMemory` —
  an operator-authored markdown document, injected whole into the durable
  system prompt.
- :class:`~langchain_agentkit.extensions.memory.recalled.RecalledMemory` —
  a LangGraph ``BaseStore`` collection the agent writes and searches
  itself.

Every method has a no-op default, so a strategy overrides only the parts
it uses.  The two content methods map onto the two prompt channels
described in :meth:`Extension.prompt`:

- :meth:`body` — bytes that stay stable across model calls, so they belong
  in the cacheable durable region.
- :meth:`reminder` — content that changes as the conversation moves, so it
  rides the tail of the conversation instead of busting the prompt cache.
"""

from __future__ import annotations

from typing import Any


class MemoryStrategy:
    """Base class for memory sources.

    Subclass and override only what the source needs.  A read-only
    document strategy implements :meth:`body`; a store-backed strategy
    implements :attr:`tools` and :meth:`reminder`.
    """

    @property
    def tools(self) -> list[Any]:
        """Tools this strategy contributes to the agent."""
        return []

    async def setup(self) -> None:
        """Run one-time async initialization.

        Called from the extension's ``setup()``, after dependency
        resolution and before the graph is built.  Exceptions propagate
        and are fatal — a mis-configured memory source should block kit
        construction rather than silently contribute nothing.
        """

    async def refresh(self, *, state: dict[str, Any], runtime: Any) -> None:
        """Refresh cached content ahead of a model call.

        Called from ``before_model``, so it runs on **every** model call —
        several times per user turn in a tool-calling loop.  Implementations
        that do real work should cache and skip when their inputs have not
        changed.
        """

    async def after_turn(self, *, state: dict[str, Any], runtime: Any) -> None:
        """React to a completed turn.

        Called from ``after_run``, once per ``ainvoke`` rather than once per
        model call.  Used for write-back that should observe the whole turn.
        """

    def body(self) -> str | None:
        """Durable prompt content, or ``None`` to contribute nothing."""
        return None

    def reminder(self) -> str | None:
        """Per-turn reminder content, or ``None`` to contribute nothing."""
        return None
