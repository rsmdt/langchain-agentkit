"""WebSearch tool over the typed provider contract.

Two execution modes, both returning ``content_and_artifact``:

- **Fallback (default)**: query providers in order, first non-empty wins.
  With one provider — the common case — that is just that provider. No
  cross-provider duplication because only one provider's results are used.
- **Fan-out (opt-in)**: query all providers concurrently, then merge, dedupe
  by URL, and rank into one list. Costs one backend call per provider, so it
  is opt-in rather than the default.

The rendered string is what the model reads; the artifact carries the
``SearchResult`` records so downstream hooks can dedupe, bound, or cite them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, override

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_agentkit.extensions.web.provider import SearchProvider, SearchResult

_WEB_SEARCH_DESCRIPTION = """Search the web for current information. Use when you need facts beyond your training or about recent events. Returns results with their source URLs."""

_DEFAULT_MAX_RESULTS = 5


class _WebSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    max_results: int = Field(
        default=_DEFAULT_MAX_RESULTS,
        ge=1,
        le=25,
        description="Maximum number of results to return.",
    )


def _rank(results: list[SearchResult]) -> list[SearchResult]:
    """Order by descending backend score; scoreless results keep input order.

    ``sorted`` is stable, so results a backend did not score fall back to the
    order the providers returned them in.
    """
    return sorted(results, key=lambda r: (r.score is not None, r.score or 0.0), reverse=True)


def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
    """Drop later results whose URL was already seen, keeping the first."""
    seen: set[str] = set()
    kept: list[SearchResult] = []
    for r in results:
        if r.url in seen:
            continue
        seen.add(r.url)
        kept.append(r)
    return kept


def _render(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    lines: list[str] = []
    for r in results:
        title = r.title or r.url
        line = f"- [{title}]({r.url})"
        if r.snippet:
            line += f": {r.snippet}"
        if r.published:
            line += f" ({r.published})"
        lines.append(line)
    return "\n".join(lines)


class _WebSearchTool(BaseTool):
    """Search tool executing providers in fallback or fan-out mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "WebSearch"
    description: str = _WEB_SEARCH_DESCRIPTION
    args_schema: type[BaseModel] = _WebSearchInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    providers: list[SearchProvider]
    fan_out: bool = False

    @override
    async def _arun(
        self, query: str, max_results: int = _DEFAULT_MAX_RESULTS
    ) -> tuple[str, list[dict[str, Any]]]:
        results = (
            await self._fan_out(query, max_results)
            if self.fan_out
            else await self._fallback(query, max_results)
        )
        return _render(results), [r.as_dict() for r in results]

    async def _fallback(self, query: str, max_results: int) -> list[SearchResult]:
        """Try providers in order; first non-empty wins. Raise only if all fail."""
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                results = await provider.asearch(query, max_results=max_results)
            except Exception as e:  # noqa: BLE001 — recorded, re-raised below if terminal
                last_error = e
                continue
            if results:
                return self._stamp(provider.name, results)[:max_results]
        if last_error is not None:
            # Every attempted provider raised: propagate so ResilienceExtension
            # (if composed) can retry or synthesize a repair message.
            raise last_error
        return []

    async def _fan_out(self, query: str, max_results: int) -> list[SearchResult]:
        """Query all providers concurrently; merge, dedupe, rank. Raise if all fail."""
        settled = await asyncio.gather(
            *(self._call(p, query, max_results) for p in self.providers),
            return_exceptions=True,
        )
        merged: list[SearchResult] = []
        errors: list[BaseException] = []
        for outcome in settled:
            if isinstance(outcome, BaseException):
                errors.append(outcome)
            else:
                merged.extend(outcome)
        if not merged and errors:
            raise errors[0]
        return _dedupe(_rank(merged))[:max_results]

    async def _call(
        self, provider: SearchProvider, query: str, max_results: int
    ) -> list[SearchResult]:
        return self._stamp(provider.name, await provider.asearch(query, max_results=max_results))

    @staticmethod
    def _stamp(name: str, results: list[SearchResult]) -> list[SearchResult]:
        """Attribute each result to the provider that returned it."""
        return [
            r if r.provider else SearchResult(**{**r.as_dict(), "provider": name}) for r in results
        ]

    @override
    def _run(
        self, query: str, max_results: int = _DEFAULT_MAX_RESULTS
    ) -> tuple[str, list[dict[str, Any]]]:
        results: list[SearchResult] = []
        last_error: Exception | None = None
        errors = 0
        for provider in self.providers:
            try:
                got = self._stamp(provider.name, provider.search(query, max_results=max_results))
            except Exception as e:  # noqa: BLE001 — recorded, re-raised below if terminal
                last_error = e
                errors += 1
                continue
            if got and not self.fan_out:
                return _render(got[:max_results]), [r.as_dict() for r in got[:max_results]]
            results.extend(got)
        if not results and errors == len(self.providers) and last_error is not None:
            raise last_error
        final = _dedupe(_rank(results))[:max_results] if self.fan_out else results[:max_results]
        return _render(final), [r.as_dict() for r in final]
