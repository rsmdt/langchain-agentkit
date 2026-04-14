"""Context compaction — evict old tool results from the LLM context window."""

from langchain_agentkit.extensions.context_compaction.extension import (
    ContextCompactionExtension,
)

__all__ = ["ContextCompactionExtension"]
