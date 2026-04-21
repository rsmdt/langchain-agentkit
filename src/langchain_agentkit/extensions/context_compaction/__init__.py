"""Context compaction — summarize old history into a synthetic checkpoint."""

from langchain_agentkit.extensions.context_compaction.extension import (
    DEFAULT_COMPACTION_SETTINGS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_RESERVE_TOKENS,
    CompactionSettings,
    ContextCompactionExtension,
)

__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_KEEP_RECENT_TOKENS",
    "DEFAULT_RESERVE_TOKENS",
    "CompactionSettings",
    "ContextCompactionExtension",
]
