"""WebSearchExtension — fan-out web search across multiple providers.

Usage::

    from langchain_agentkit import WebSearchExtension, QwantSearchProvider

    mw = WebSearchExtension()  # defaults to built-in Qwant provider
    mw = WebSearchExtension(providers=[my_search_tool, another_search_fn])

    # Use QwantSearchProvider standalone as any other LangChain tool
    tool = QwantSearchProvider()
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_web_search_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "web_search_system.md")


def _default_user_agent() -> str:
    """Build a User-Agent string from the host machine's platform info."""
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    py = platform.python_version()
    return f"langchain-agentkit/Python {py} ({system} {release}; {machine})"


class DuckDuckGoSearchProvider(BaseTool):
    """Built-in web search using DuckDuckGo's instant answer API.

    No API key required. Returns abstracts, related topics, and
    direct answers from DuckDuckGo's public API.

    Args:
        max_results: Maximum number of related topics to return.
        headers: HTTP headers for requests. Defaults to a User-Agent
            derived from the host machine's platform info.
    """

    name: str = "DuckDuckGoSearch"
    description: str = "Search the web using DuckDuckGo."
    max_results: int = 5
    headers: dict[str, str] | None = None

    def __init__(self, **kwargs: Any) -> None:
        if "headers" not in kwargs or kwargs["headers"] is None:
            kwargs["headers"] = {"User-Agent": _default_user_agent()}
        super().__init__(**kwargs)

    def _run(self, query: str) -> str:
        """Sync search via DuckDuckGo instant answer API."""
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = urllib.request.Request(url, headers=self.headers)
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

    async def _arun(self, query: str) -> str:
        """Async version — runs sync in executor (urllib has no async)."""
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

    def _run(self, query: str) -> str:
        """Sync search via Qwant API."""
        params = urllib.parse.urlencode(
            {
                "q": query,
                "count": self.max_results,
                "locale": self.locale,
                "safesearch": self.safesearch,
            },
        )
        url = f"https://api.qwant.com/v3/search/web?{params}"
        request = urllib.request.Request(url, headers=self.headers)
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

    async def _arun(self, query: str) -> str:
        """Async version — runs sync in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)



class _WebSearchTool(BaseTool):
    """Internal fan-out tool that calls all providers concurrently."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "WebSearch"
    description: str = "Search the web using multiple search providers simultaneously."
    providers: list[BaseTool]

    async def _arun(self, query: str) -> str:
        """Fan out to all providers concurrently, capturing errors per-provider."""
        results = await asyncio.gather(*[self._call_provider(p, query) for p in self.providers])
        return "\n\n".join(f"## {name}\n{result}" for name, result in results)

    async def _call_provider(self, provider: BaseTool, query: str) -> tuple[str, str]:
        try:
            result = await provider.ainvoke(query)
            return (provider.name, str(result))
        except Exception as e:
            return (provider.name, f"Error: {e}")

    def _run(self, query: str) -> str:
        """Sync fallback — sequential provider calls."""
        sections = []
        for provider in self.providers:
            try:
                result = provider.invoke(query)
                sections.append(f"## {provider.name}\n{result}")
            except Exception as e:
                sections.append(f"## {provider.name}\nError: {e}")
        return "\n\n".join(sections)


class WebSearchExtension(Extension):
    """Extension providing a single WebSearch tool that fans out to
    multiple configured search providers in parallel.

    Each provider is a BaseTool or callable supplied by the application.
    The extensions creates a single ``WebSearch`` tool that:
    1. Calls all providers concurrently via asyncio.gather
    2. Returns results attributed per provider

    Args:
        providers: List of search providers. Each is a BaseTool instance
            or a callable with signature ``(query: str) -> str``.
            Callables are auto-wrapped into BaseTool via @tool.
            When ``None`` or empty, defaults to the built-in Qwant provider.
        prompt_template: Optional custom prompt template path or string.
            Defaults to built-in search guidance.

    Example::

        mw = WebSearchExtension()  # uses built-in Qwant provider
        mw = WebSearchExtension(providers=[my_search_tool])
        mw.tools   # [WebSearch]
        mw.prompt(state, runtime)  # Search guidance with provider names
    """

    def __init__(
        self,
        providers: list[BaseTool | Callable[[str], str]] | None = None,
        prompt_template: str | Path | None = None,
    ) -> None:
        if providers is None or len(providers) == 0:
            providers = [QwantSearchProvider()]

        self._providers = [self._resolve_provider(p) for p in providers]
        self._prompt_template = self._load_prompt(prompt_template)
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

    def _load_prompt(self, prompt_template: str | Path | None) -> PromptTemplate:
        if prompt_template is None:
            return _web_search_system_prompt
        path = Path(prompt_template)
        if path.exists():
            return PromptTemplate.from_file(path)
        return PromptTemplate.from_template(str(prompt_template))

    @property
    def state_schema(self) -> None:
        """No additional state keys."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """Returns [WebSearch] — same list object on every access."""
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime) -> str | None:
        """Search guidance listing configured provider names."""
        provider_names = ", ".join(p.name for p in self._providers)
        return self._prompt_template.format(provider_names=provider_names)
