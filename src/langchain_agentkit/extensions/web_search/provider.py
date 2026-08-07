"""Search provider contract — typed result records.

Providers return records, not prose. Structure is what lets the search layer
dedupe by URL, rank across providers, bound what enters context, and expose
the returned URL set for citation checking. A provider that flattens to a
string destroys all four.

Usage::

    class MySearchProvider(SearchProvider):
        name = "MySearch"

        def search(self, query: str, *, max_results: int) -> list[SearchResult]:
            hits = my_api.query(query, limit=max_results)
            return [SearchResult(url=h.url, title=h.title, snippet=h.text) for h in hits]
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One web search hit.

    Args:
        url: Result URL. The identity field — dedupe and citation checking key on it.
        title: Result title, empty when the backend supplies none.
        snippet: Backend-supplied extract. Search backends return extracts, not
            page bodies; reading a page is a fetch concern.
        published: Publication or freshness timestamp as the backend reports it.
            ``None`` when the backend supplies none — most SERP scrapers do not.
        score: Backend relevance score, when it reports one. Used for ranking
            when merging across providers.
        provider: Name of the provider that returned this result. Set by the
            search layer, not by providers.
    """

    url: str
    title: str = ""
    snippet: str = ""
    published: str | None = None
    score: float | None = None
    provider: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form, for the tool artifact channel.

        The artifact rides in a ``ToolMessage`` and is checkpointed, so it must
        stay serializable — dataclasses are not.
        """
        return asdict(self)


class SearchProvider(ABC):
    """Base class for web search providers.

    Subclasses implement the blocking :meth:`search`. The async :meth:`asearch`
    defaults to running it in an executor; override it directly when the backend
    has a native async client.
    """

    name: str = "search"

    @abstractmethod
    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Run ``query`` and return at most ``max_results`` records."""

    async def asearch(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Async form of :meth:`search`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self.search, query, max_results=max_results),
        )
