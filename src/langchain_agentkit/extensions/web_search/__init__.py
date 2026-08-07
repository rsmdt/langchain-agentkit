"""Web extensions — provider-backed search (typed records) and URL fetch."""

from langchain_agentkit.extensions.web_search.extension import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebSearchExtension,
)
from langchain_agentkit.extensions.web_search.provider import SearchProvider, SearchResult

__all__ = [
    "DuckDuckGoSearchProvider",
    "QwantSearchProvider",
    "SearchProvider",
    "SearchResult",
    "WebSearchExtension",
]
