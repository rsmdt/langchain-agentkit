"""Backward-compatible re-exports from ``langchain_agentkit.backends``.

.. deprecated::
    Import from ``langchain_agentkit.backends`` instead.
    This module will be removed in a future version.

The canonical location for backend types is now
``langchain_agentkit.backends.protocol`` and
``langchain_agentkit.backends.os``.
"""

# Re-export everything from the new locations so existing imports still work.
from langchain_agentkit.backends.os import OSBackend as OSBackend
from langchain_agentkit.backends.protocol import (
    BackendProtocol as BackendProtocol,
    EditResult as EditResult,
    ExecuteResponse as ExecuteResponse,
    FileInfo as FileInfo,
    GrepMatch as GrepMatch,
    WriteResult as WriteResult,
)
