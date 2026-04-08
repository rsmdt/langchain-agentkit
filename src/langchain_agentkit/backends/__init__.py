"""Backend implementations for agent environments.

Two backends are provided:

- ``OSBackend`` — native local filesystem via Python stdlib.
- ``DaytonaBackend`` — Daytona cloud sandbox via shell commands.

Both implement the ``BackendProtocol`` which mirrors the Claude Code
tool surface: Read, Write, Edit, Glob, Grep, Bash.

Usage::

    from langchain_agentkit.backends import BackendProtocol, OSBackend, DaytonaBackend
"""

from langchain_agentkit.backends.daytona import DaytonaBackend
from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileInfo,
    GrepMatch,
    WriteResult,
)

__all__ = [
    "BackendProtocol",
    "DaytonaBackend",
    "EditResult",
    "ExecuteResponse",
    "FileInfo",
    "GrepMatch",
    "OSBackend",
    "WriteResult",
]
