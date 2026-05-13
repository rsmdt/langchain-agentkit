"""History extension — rewrite ``state["messages"]`` via a pluggable strategy.

See :class:`HistoryExtension` for the overall mechanism and the three
built-in strategies (:class:`CountStrategy`, :class:`TokenStrategy`,
:class:`CompactionStrategy`).
"""

from langchain_agentkit.extensions.history.compaction import (
    DEFAULT_CONTEXT_WINDOW as DEFAULT_CONTEXT_WINDOW,
)
from langchain_agentkit.extensions.history.compaction import (
    DEFAULT_RESERVE_TOKENS as DEFAULT_RESERVE_TOKENS,
)
from langchain_agentkit.extensions.history.compaction import (
    CompactionStrategy as CompactionStrategy,
)
from langchain_agentkit.extensions.history.extension import (
    HistoryExtension as HistoryExtension,
)
from langchain_agentkit.extensions.history.state import (
    ReplaceMessages as ReplaceMessages,
)
from langchain_agentkit.extensions.history.strategies import (
    CountStrategy as CountStrategy,
)
from langchain_agentkit.extensions.history.strategies import (
    HistoryStrategy as HistoryStrategy,
)
from langchain_agentkit.extensions.history.strategies import (
    TokenStrategy as TokenStrategy,
)

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_RESERVE_TOKENS",
    "CompactionStrategy",
    "CountStrategy",
    "HistoryExtension",
    "HistoryStrategy",
    "ReplaceMessages",
    "TokenStrategy",
]
