"""Backend implementations for agent environments.

The 6-method ``BackendProtocol`` mirrors the Claude Code tool surface:
Read, Write, Edit, Glob, Grep, Bash.

For sandbox providers with shell access, ``BaseSandbox`` implements
all file operations via shell commands — a new provider only needs
``execute()``.

Usage::

    from langchain_agentkit.backends import BackendProtocol, OSBackend, BaseSandbox
"""

from langchain_agentkit.backends.base import BaseSandbox
from langchain_agentkit.backends.daytona import DaytonaSandbox
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
    "BaseSandbox",
    "DaytonaSandbox",
    "EditResult",
    "ExecuteResponse",
    "FileInfo",
    "GrepMatch",
    "OSBackend",
    "WriteResult",
]
