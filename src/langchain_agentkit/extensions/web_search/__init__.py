"""Web search extension — fan-out web search across providers."""

from langchain_agentkit.extensions.web_search.extension import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebSearchExtension,
)

__all__ = [
    "DuckDuckGoSearchProvider",
    "QwantSearchProvider",
    "WebSearchExtension",
]
