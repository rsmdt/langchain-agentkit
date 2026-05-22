"""WebSearch fan-out tool that calls multiple providers concurrently."""

from __future__ import annotations

import asyncio
from typing import override

from langchain_core.tools import BaseTool
from pydantic import ConfigDict

_WEB_SEARCH_DESCRIPTION = """Search the web for up-to-date information beyond the model's knowledge cutoff. Returns results as markdown blocks with hyperlinks; searching happens within a single call.
MANDATORY: end your response with a "Sources:" section listing the relevant result URLs as markdown links ([Title](URL)). Never skip it.
- Domain filtering depends on the configured provider.
- Use the current year when searching for recent information."""


class _WebSearchTool(BaseTool):
    """Internal fan-out tool that calls all providers concurrently."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "WebSearch"
    description: str = _WEB_SEARCH_DESCRIPTION
    providers: list[BaseTool]

    @override
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

    @override
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
