"""MemoryExtension — surface persistent memory to the agent each turn.

The extension owns *framing* — a header, a staleness caveat, and any
caller-supplied extra sources — and delegates *sourcing* to a
:class:`MemoryStrategy`.

The default strategy,
:class:`~langchain_agentkit.extensions.memory.curated.CuratedMemory`,
injects an operator-authored markdown document into the durable system
prompt::

    MemoryExtension()  # ~/.agents/memory/<project-key>/MEMORY.md

:class:`~langchain_agentkit.extensions.memory.recalled.RecalledMemory`
instead gives the agent a store it writes and searches itself::

    MemoryExtension(strategy=RecalledMemory(namespace=("memories",)))

The two compose by declaring the extension twice — a curated document that
is always present, alongside a learned store the agent queries on demand.

Content routes to whichever prompt channel suits it: a strategy's
``body()`` joins the cacheable durable region, while its ``reminder()``
rides the tail of the conversation. The staleness caveat is stated once,
on whichever channel carries content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.memory.curated import CuratedMemory

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_agentkit.extensions.memory.strategy import MemoryStrategy

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MemoryExtension(Extension):
    """Contribute persistent memory to the agent's prompt.

    Args:
        strategy: Where memory comes from. Defaults to a
            :class:`CuratedMemory` document.
        header: Heading for the durable memory section.
        trust_footer: Staleness caveat, emitted once alongside whichever
            channel carries memory content.
        extra_sources: Callables contributing additional durable text.
            A source that returns ``None``/``""`` or raises is skipped.
    """

    strategy: MemoryStrategy = field(default_factory=CuratedMemory)
    header: str = "# Memory"
    trust_footer: str = (
        "Memory records may be stale. Verify facts against current state before acting on them."
    )
    extra_sources: list[Callable[[], str | None]] = field(default_factory=list)

    # --- Extension protocol ---

    @property
    @override
    def tools(self) -> list[Any]:
        return self.strategy.tools

    @override
    async def setup(self, **_: Any) -> None:  # type: ignore[override]
        await self.strategy.setup()

    async def before_model(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        await self.strategy.refresh(state=state, runtime=runtime)
        return None

    async def after_run(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        await self.strategy.after_turn(state=state, runtime=runtime)
        return None

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: Any | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> dict[str, str] | None:
        durable = self._compose_durable()
        reminder = self.strategy.reminder()

        if durable is None and reminder is None:
            return None

        out: dict[str, str] = {}
        if durable is not None:
            out["prompt"] = durable
        if reminder is not None:
            # The caveat belongs with memory content wherever it lands, but
            # only once — the durable section already carries it when present.
            if durable is None:
                reminder = reminder + "\n\n" + self.trust_footer
            out["reminder"] = reminder
        return out

    # --- Internals ---

    def _compose_durable(self) -> str | None:
        body = self.strategy.body()
        extras = self._collect_extras()

        if body is None and not extras:
            return None

        sections: list[str] = [self.header, ""]
        if body is not None:
            sections.append(body)
        if extras:
            if body is not None:
                sections.append("")
            sections.append("\n\n".join(extras))
        sections.append("")
        sections.append(self.trust_footer)
        return "\n".join(sections)

    def _collect_extras(self) -> list[str]:
        out: list[str] = []
        for source in self.extra_sources:
            try:
                value = source()
            except Exception:
                logger.debug(
                    "MemoryExtension: extra_source raised; skipping",
                    exc_info=True,
                )
                continue
            if value is None or value == "":
                continue
            out.append(value)
        return out
