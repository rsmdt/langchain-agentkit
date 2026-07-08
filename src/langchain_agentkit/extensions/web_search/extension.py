"""WebSearchExtension — fan-out web search across multiple providers.

Usage::

    from langchain_agentkit import WebSearchExtension, DuckDuckGoSearchProvider

    mw = WebSearchExtension()  # defaults to built-in DuckDuckGo provider
    mw = WebSearchExtension(providers=[my_search_tool, another_search_fn])

    # Use DuckDuckGoSearchProvider standalone as any other LangChain tool
    tool = DuckDuckGoSearchProvider()                # instant-answer mode
    tool = DuckDuckGoSearchProvider(use_html=True)   # opt-in general web results
    result = tool.invoke("latest AI news")
    tools = mw.tools           # [WebSearch]
    prompt = mw.prompt(state, runtime)  # Search guidance with provider names
"""

from __future__ import annotations

import asyncio
import json
import platform
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, override

from langchain_core.tools import BaseTool

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.web_search.tools.web_search import _WebSearchTool

if TYPE_CHECKING:
    from collections.abc import Callable


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


class DuckDuckGoSearchProvider(BaseTool):
    """Built-in web search using DuckDuckGo. No API key required.

    Two modes:

    - Default (``use_html=False``): the Instant Answer API. Returns
      abstracts, related topics, and direct answers. Reliable, but only
      covers queries with an instant-answer entry — many general queries
      return "No results found."
    - Opt-in (``use_html=True``): the HTML SERP endpoint. Returns ranked
      general web results scraped from DuckDuckGo's HTML interface. Broader
      coverage, but scrapes markup and may rate-limit or change shape.

    Args:
        max_results: Maximum number of results to return.
        use_html: When True, query the HTML SERP endpoint for general web
            results instead of the Instant Answer API.
        headers: HTTP headers for requests. Defaults to a User-Agent
            derived from the host machine's platform info, or a browser
            User-Agent in HTML mode (the HTML endpoint blocks non-browser
            clients).
    """

    name: str = "DuckDuckGoSearch"
    description: str = "Search the web using DuckDuckGo."
    max_results: int = 5
    use_html: bool = False
    headers: dict[str, str] | None = None

    def __init__(self, **kwargs: Any) -> None:
        if "headers" not in kwargs or kwargs["headers"] is None:
            user_agent = _BROWSER_USER_AGENT if kwargs.get("use_html") else _default_user_agent()
            kwargs["headers"] = {"User-Agent": user_agent}
        super().__init__(**kwargs)

    @override
    def _run(self, query: str) -> str:
        if self.use_html:
            return self._run_html(query)
        return self._run_instant(query)

    def _run_instant(self, query: str) -> str:
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = urllib.request.Request(url, headers=self.headers)  # type: ignore[arg-type]
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())

        results: list[str] = []

        # Abstract (main answer)
        abstract = data.get("Abstract", "")
        abstract_url = data.get("AbstractURL", "")
        if abstract:
            source = data.get("AbstractSource", "")
            results.append(f"**{source}**: {abstract}")
            if abstract_url:
                results.append(f"  Source: {abstract_url}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[: self.max_results]:
            text = topic.get("Text", "")
            first_url = topic.get("FirstURL", "")
            if text:
                results.append(f"- {text}")
                if first_url:
                    results[-1] += f" ({first_url})"

        if not results:
            return "No results found."
        return "\n".join(results)

    def _run_html(self, query: str) -> str:
        params = urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            "https://html.duckduckgo.com/html/",
            data=params.encode(),
            headers=self.headers,  # type: ignore[arg-type]
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", "replace")

        parser = _DDGHTMLResultParser()
        parser.feed(body)
        if not parser.results:
            return "No results found."
        return "\n".join(
            f"- [{r['title']}]({r['url']}): {r['snippet']}"
            for r in parser.results[: self.max_results]
        )

    @override
    async def _arun(self, query: str) -> str:
        # Runs sync in executor — urllib has no async API.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)


class QwantSearchProvider(BaseTool):
    """Web search provider using Qwant's API. No API key required.

    Note: Qwant's API may block requests depending on region or rate
    limits. Use :class:`DuckDuckGoSearchProvider` as a more reliable
    alternative.

    Args:
        max_results: Maximum number of results to return.
        locale: Search locale (e.g., ``"en_US"``, ``"fr_FR"``).
        safesearch: Safe search level: 0=off, 1=moderate, 2=strict.
        headers: HTTP headers for requests.
    """

    name: str = "QwantSearch"
    description: str = "Search the web using Qwant."
    max_results: int = 5
    locale: str = "en_US"
    safesearch: int = 1
    headers: dict[str, str] | None = None

    def __init__(self, **kwargs: Any) -> None:
        if "headers" not in kwargs or kwargs["headers"] is None:
            kwargs["headers"] = {"User-Agent": _default_user_agent()}
        super().__init__(**kwargs)

    @override
    def _run(self, query: str) -> str:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": self.max_results,
                "locale": self.locale,
                "safesearch": self.safesearch,
            },
        )
        url = f"https://api.qwant.com/v3/search/web?{params}"
        request = urllib.request.Request(url, headers=self.headers)  # type: ignore[arg-type]
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
        items = data.get("data", {}).get("result", {}).get("items", [])
        if not items:
            return "No results found."
        results = []
        for item in items[: self.max_results]:
            title = item.get("title", "")
            item_url = item.get("url", "")
            snippet = item.get("desc", "")
            results.append(f"- [{title}]({item_url}): {snippet}")
        return "\n".join(results)

    @override
    async def _arun(self, query: str) -> str:
        # Runs sync in executor — urllib has no async API.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)


class WebSearchExtension(Extension):
    """Extension providing a single WebSearch tool that fans out to
    multiple configured search providers in parallel.

    Each provider is a BaseTool or callable supplied by the application.
    The extension creates a single ``WebSearch`` tool that:
    1. Calls all providers concurrently via asyncio.gather
    2. Returns results attributed per provider

    Args:
        providers: List of search providers. Each is a BaseTool instance
            or a callable with signature ``(query: str) -> str``.
            Callables are auto-wrapped into a ``StructuredTool``.
            When ``None`` or empty, defaults to the built-in DuckDuckGo provider.

    Example::

        mw = WebSearchExtension()  # uses built-in DuckDuckGo provider
        mw = WebSearchExtension(providers=[my_search_tool])
        mw.tools   # [WebSearch]
        mw.prompt(state, runtime)  # Search guidance with provider names
    """

    def __init__(
        self,
        *,
        providers: list[BaseTool | Callable[[str], str]] | None = None,
    ) -> None:
        if providers is None or len(providers) == 0:
            providers = [DuckDuckGoSearchProvider()]

        self._providers = [self._resolve_provider(p) for p in providers]
        self._tools: list[BaseTool] = [_WebSearchTool(providers=self._providers)]

    def _resolve_provider(self, provider: BaseTool | Callable[[str], str]) -> BaseTool:
        if isinstance(provider, BaseTool):
            return provider
        if callable(provider):
            from langchain_core.tools.structured import StructuredTool

            description = getattr(provider, "__doc__", None) or "Search provider."
            return StructuredTool.from_function(
                func=provider,
                description=description,
            )
        raise TypeError(
            f"Each provider must be a BaseTool or callable, got {type(provider).__name__!r}"
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
        # The behavioral norm lives here; per-call selection guidance lives
        # on the ``WebSearch`` tool description.
        return (
            'After answering, end with a "Sources:" section listing the '
            "result URLs as markdown links."
        )
