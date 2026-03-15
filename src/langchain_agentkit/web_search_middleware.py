"""WebSearchMiddleware — fan-out web search across multiple providers.

Usage::

    from langchain_agentkit import WebSearchMiddleware, QwantSearchTool

    mw = WebSearchMiddleware()  # defaults to built-in Qwant provider
    mw = WebSearchMiddleware(providers=[my_search_tool, another_search_fn])

    # Use QwantSearchTool standalone as any other LangChain tool
    tool = QwantSearchTool()
    result = tool.invoke("latest AI news")
    tools = mw.tools           # [web_search]
    prompt = mw.prompt(state, runtime)  # Search guidance with provider names
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_web_search_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "web_search_system.md")


class QwantSearchTool(BaseTool):
    """Built-in web search provider using Qwant's API. No API key required."""

    name: str = "qwant_search"
    description: str = "Search the web using Qwant."
    max_results: int = 5
    locale: str = "en_US"
    safesearch: int = 1  # 0=off, 1=moderate, 2=strict

    def _run(self, query: str, **kwargs: Any) -> str:
        """Sync search via Qwant API using urllib (no external deps)."""
        params = urllib.parse.urlencode({
            "q": query,
            "count": self.max_results,
            "locale": self.locale,
            "safesearch": self.safesearch,
        })
        url = f"https://api.qwant.com/v3/search/web?{params}"
        request = urllib.request.Request(url, headers={
            "User-Agent": "langchain-agentkit/0.5",
        })
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
        items = data.get("data", {}).get("result", {}).get("items", [])
        if not items:
            return "No results found."
        results = []
        for item in items[:self.max_results]:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("desc", "")
            results.append(f"- [{title}]({url}): {snippet}")
        return "\n".join(results)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Async version — runs sync in executor (urllib has no async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)


class _WebSearchTool(BaseTool):
    """Internal fan-out tool that calls all providers concurrently."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_search"
    description: str = "Search the web using multiple search providers simultaneously."
    providers: list[BaseTool]

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """Fan out to all providers concurrently, capturing errors per-provider."""
        results = await asyncio.gather(*[self._call_provider(p, query) for p in self.providers])
        return "\n\n".join(f"## {name}\n{result}" for name, result in results)

    async def _call_provider(self, provider: BaseTool, query: str) -> tuple[str, str]:
        try:
            result = await provider.ainvoke(query)
            return (provider.name, str(result))
        except Exception as e:
            return (provider.name, f"Error: {e}")

    def _run(self, query: str, **kwargs: Any) -> str:
        """Sync fallback — sequential provider calls."""
        sections = []
        for provider in self.providers:
            try:
                result = provider.invoke(query)
                sections.append(f"## {provider.name}\n{result}")
            except Exception as e:
                sections.append(f"## {provider.name}\nError: {e}")
        return "\n\n".join(sections)


class WebSearchMiddleware:
    """Middleware providing a single web_search tool that fans out to
    multiple configured search providers in parallel.

    Each provider is a BaseTool or callable supplied by the application.
    The middleware creates a single ``web_search`` tool that:
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

        mw = WebSearchMiddleware()  # uses built-in Qwant provider
        mw = WebSearchMiddleware(providers=[my_search_tool])
        mw.tools   # [web_search]
        mw.prompt(state, runtime)  # Search guidance with provider names
    """

    def __init__(
        self,
        providers: list[BaseTool | Callable[[str], str]] | None = None,
        prompt_template: str | Path | None = None,
    ) -> None:
        if providers is None or len(providers) == 0:
            providers = [QwantSearchTool()]

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
    def tools(self) -> list[BaseTool]:
        """Returns [web_search] — same list object on every access."""
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime) -> str | None:
        """Search guidance listing configured provider names."""
        provider_names = ", ".join(p.name for p in self._providers)
        return self._prompt_template.format(provider_names=provider_names)
