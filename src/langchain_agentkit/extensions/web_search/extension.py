"""WebSearchExtension — web search over a typed provider contract.

Usage::

    from langchain_agentkit import WebSearchExtension, DuckDuckGoSearchProvider

    ext = WebSearchExtension()                       # built-in DuckDuckGo provider
    ext = WebSearchExtension(providers=[my_provider])
    ext = WebSearchExtension(providers=[a, b], fan_out=True)  # concurrent, merged

    # A provider is any SearchProvider; use one standalone:
    provider = DuckDuckGoSearchProvider()            # instant-answer mode
    provider = DuckDuckGoSearchProvider(use_html=True)   # general web results
    records = provider.search("latest AI news", max_results=5)  # list[SearchResult]
"""

from __future__ import annotations

import json
import platform
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.web_search.provider import SearchProvider, SearchResult
from langchain_agentkit.extensions.web_search.tools.web_search import _WebSearchTool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _default_user_agent() -> str:
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    py = platform.python_version()
    return f"langchain-agentkit/Python {py} ({system} {release}; {machine})"


# DuckDuckGo's HTML SERP endpoint soft-blocks non-browser clients (HTTP 202,
# empty body), so HTML mode must present a browser User-Agent. The Instant
# Answer API has no such requirement.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class _DDGHTMLResultParser(HTMLParser):
    """Extract (title, url, snippet) from DuckDuckGo's HTML SERP markup.

    Results are ``<a class="result__a" href=...>title</a>`` for the title and
    URL, followed by ``<a class="result__snippet">text</a>`` for the snippet.
    A result is committed once its snippet anchor closes.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._buf: list[str] = []
        self._current: dict[str, str] | None = None

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        classes = (dict(attrs).get("class") or "").split()
        if "result__a" in classes:
            self._current = {"url": dict(attrs).get("href") or "", "title": "", "snippet": ""}
            self._capture = "title"
            self._buf = []
        elif "result__snippet" in classes and self._current is not None:
            self._capture = "snippet"
            self._buf = []

    @override
    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._buf.append(data)

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._capture is None or self._current is None:
            return
        text = "".join(self._buf).strip()
        if self._capture == "title":
            self._current["title"] = text
        else:
            self._current["snippet"] = text
            self.results.append(self._current)
            self._current = None
        self._capture = None
        self._buf = []


class DuckDuckGoSearchProvider(SearchProvider):
    """Built-in web search using DuckDuckGo. No API key required.

    Two modes:

    - Default (``use_html=False``): the Instant Answer API. Returns
      abstracts, related topics, and direct answers. Reliable, but only
      covers queries with an instant-answer entry — many general queries
      return no results.
    - Opt-in (``use_html=True``): the HTML SERP endpoint. Returns general
      web results scraped from DuckDuckGo's HTML interface. Broader
      coverage, but scrapes markup and may rate-limit or change shape.

    Neither mode returns publication timestamps, so ``SearchResult.published``
    is always ``None`` here.

    Args:
        max_results: Default result cap when a caller does not pass one.
        use_html: When True, query the HTML SERP endpoint for general web
            results instead of the Instant Answer API.
        headers: HTTP headers for requests. Defaults to a User-Agent
            derived from the host machine's platform info, or a browser
            User-Agent in HTML mode.
    """

    name = "DuckDuckGoSearch"

    def __init__(
        self,
        *,
        max_results: int = 5,
        use_html: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.max_results = max_results
        self.use_html = use_html
        if headers is None:
            headers = {"User-Agent": _BROWSER_USER_AGENT if use_html else _default_user_agent()}
        self.headers = headers

    @override
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        limit = max_results if max_results is not None else self.max_results
        if self.use_html:
            return self._search_html(query, limit)
        return self._search_instant(query, limit)

    def _search_instant(self, query: str, limit: int) -> list[SearchResult]:
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())

        results: list[SearchResult] = []
        abstract = data.get("Abstract", "")
        abstract_url = data.get("AbstractURL", "")
        if abstract and abstract_url:
            results.append(
                SearchResult(
                    url=abstract_url,
                    title=data.get("AbstractSource", "") or abstract_url,
                    snippet=abstract,
                )
            )
        for topic in data.get("RelatedTopics", []):
            text = topic.get("Text", "")
            first_url = topic.get("FirstURL", "")
            if text and first_url:
                results.append(SearchResult(url=first_url, title=text[:80], snippet=text))
            if len(results) >= limit:
                break
        return results[:limit]

    def _search_html(self, query: str, limit: int) -> list[SearchResult]:
        params = urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=params.encode(),
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")

        parser = _DDGHTMLResultParser()
        parser.feed(body)
        return [
            SearchResult(url=r["url"], title=r["title"], snippet=r["snippet"])
            for r in parser.results[:limit]
        ]


class QwantSearchProvider(SearchProvider):
    """Web search provider using Qwant's API. No API key required.

    Note: Qwant's API may block requests depending on region or rate
    limits. Use :class:`DuckDuckGoSearchProvider` as a more reliable
    alternative.

    Args:
        max_results: Default result cap when a caller does not pass one.
        locale: Search locale (e.g., ``"en_US"``, ``"fr_FR"``).
        safesearch: Safe search level: 0=off, 1=moderate, 2=strict.
        headers: HTTP headers for requests.
    """

    name = "QwantSearch"

    def __init__(
        self,
        *,
        max_results: int = 5,
        locale: str = "en_US",
        safesearch: int = 1,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.max_results = max_results
        self.locale = locale
        self.safesearch = safesearch
        self.headers = headers or {"User-Agent": _default_user_agent()}

    @override
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        limit = max_results if max_results is not None else self.max_results
        params = urllib.parse.urlencode(
            {"q": query, "count": limit, "locale": self.locale, "safesearch": self.safesearch},
        )
        url = f"https://api.qwant.com/v3/search/web?{params}"
        request = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
        items = data.get("data", {}).get("result", {}).get("items", [])
        return [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("desc", ""),
            )
            for item in items[:limit]
            if item.get("url")
        ]


class WebSearchExtension(Extension):
    """Extension bundling the web capability: a ``WebSearch`` tool and, when
    ``fetch=True``, a companion ``WebFetch`` tool that reads one URL as markdown.

    Search discovers URLs; fetch reads one. They travel together because the
    pair is how the model actually works: search, then read a result. Fetch is
    opt-in because it needs the ``web`` extra
    (``pip install langchain-agentkit[web]``) for HTML→markdown conversion,
    while search runs on the standard library alone.

    By default the search tool queries providers in fallback order — the first
    provider to return results wins — which with a single provider (the common
    case) is just that provider, with no cross-provider duplication. Set
    ``fan_out=True`` to query every provider concurrently and merge, dedupe,
    and rank their results into one list (one backend call per provider).

    ``WebFetch`` is SSRF-guarded unconditionally. For human approval on each
    fetch, compose ``HITLExtension(interrupt_on=["WebFetch"])``.

    Args:
        providers: List of :class:`SearchProvider` instances. When ``None`` or
            empty, defaults to the built-in DuckDuckGo provider.
        fan_out: When True, query all providers concurrently and merge results.
            When False (default), query providers in order until one returns.
        fetch: When True, also expose a ``WebFetch`` URL-reading tool. Requires
            the ``web`` extra; construction raises ``ImportError`` without it.
        fetch_timeout: Per-request timeout (seconds) for ``WebFetch``, per hop.

    Example::

        ext = WebSearchExtension()                       # WebSearch only
        ext = WebSearchExtension(providers=[my_provider])
        ext = WebSearchExtension(providers=[a, b], fan_out=True)
        ext = WebSearchExtension(fetch=True)             # WebSearch + WebFetch
    """

    def __init__(
        self,
        *,
        providers: list[SearchProvider] | None = None,
        fan_out: bool = False,
        fetch: bool = False,
        fetch_timeout: float = 30.0,
    ) -> None:
        if not providers:
            providers = [DuckDuckGoSearchProvider()]
        self._providers = [self._resolve_provider(p) for p in providers]
        self._tools: list[BaseTool] = [_WebSearchTool(providers=self._providers, fan_out=fan_out)]
        if fetch:
            self._tools.append(self._build_fetch_tool(fetch_timeout))

    def _build_fetch_tool(self, timeout: float) -> BaseTool:
        # Import lazily: the fetch tool pulls the optional `web` extra only when
        # asked for, so a search-only kit never imports the conversion path.
        # Fail fast at construction rather than at first fetch: a kit that asks
        # for fetch without the extra is misconfigured, and should say so now.
        import importlib.util

        from langchain_agentkit.extensions.web_search.tools.web_fetch import _WebFetchTool

        if importlib.util.find_spec("markdownify") is None:
            raise ImportError(
                "WebSearchExtension(fetch=True) needs the 'web' extra: "
                "pip install langchain-agentkit[web]"
            )
        return _WebFetchTool(timeout=timeout)

    def _resolve_provider(self, provider: SearchProvider) -> SearchProvider:
        if isinstance(provider, SearchProvider):
            return provider
        raise TypeError(
            f"Each provider must be a SearchProvider, got {type(provider).__name__!r}. "
            "Wrap a backend by subclassing SearchProvider and returning list[SearchResult]."
        )

    @property
    @override
    def tools(self) -> list[BaseTool]:
        return self._tools

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: Any | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str | None:
        # Only contribute the norm when the WebSearch tool is actually composed.
        # A user tool of the same name shadows ours (kit.tools dedupes user-first),
        # so emitting this unconditionally would ship guidance for a tool that is
        # not present. Empty `tools` means "called outside a composed kit" — keep
        # the norm so standalone prompt() calls still see it.
        if tools and "WebSearch" not in tools:
            return None
        return (
            'After answering, end with a "Sources:" section listing the '
            "result URLs as markdown links."
        )
