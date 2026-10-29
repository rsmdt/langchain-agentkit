"""Web extensions — provider-backed search (typed records) and URL fetch."""

from langchain_agentkit.extensions.web.extension import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebExtension,
)
from langchain_agentkit.extensions.web.provider import SearchProvider, SearchResult

__all__ = [
    "DuckDuckGoSearchProvider",
    "QwantSearchProvider",
    "SearchProvider",
    "SearchResult",
    "WebExtension",
]
