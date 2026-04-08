"""History extension — context window management."""

from langchain_agentkit.extensions.history.extension import (
    HistoryExtension as HistoryExtension,
)
from langchain_agentkit.extensions.history.state import (
    ReplaceMessages as ReplaceMessages,
)
from langchain_agentkit.extensions.history.strategies import (
    HistoryStrategy as HistoryStrategy,
)

__all__ = [
    "HistoryExtension",
    "HistoryStrategy",
    "ReplaceMessages",
]
