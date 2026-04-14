"""WebSearch fan-out tool that calls multiple providers concurrently."""

from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from pydantic import ConfigDict

_WEB_SEARCH_DESCRIPTION = """- Allows the agent to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond the model's knowledge cutoff
- Searches are performed automatically within a single call

CRITICAL REQUIREMENT - You MUST follow this:
  - After replying to the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your reply here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering support depends on the configured search provider
  - Use the current year when searching for recent information, documentation, or current events"""


class _WebSearchTool(BaseTool):
    """Internal fan-out tool that calls all providers concurrently."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "WebSearch"
    description: str = _WEB_SEARCH_DESCRIPTION
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
