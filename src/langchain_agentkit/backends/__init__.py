"""Backend implementations for agent environments.

This namespace exposes the **capability protocols and shared types**
only. Concrete backends live in their own submodules and are imported
explicitly — this keeps optional-dependency gates honest (DaytonaBackend
needs ``daytona-sdk``, future remote backends will need their own SDKs)
and surfaces the dependency at the import line.

Usage::

    # Capability protocols and shared types (always available)
    from langchain_agentkit.backends import FilesystemProtocol, SandboxProtocol

    # Concrete backends — explicit import per-backend
    from langchain_agentkit.backends.os import OSBackend
    from langchain_agentkit.backends.daytona import DaytonaBackend
"""

from langchain_agentkit.backends.helpers import read_tree
from langchain_agentkit.backends.protocol import (
    ExecuteResponse,
    FileInfo,
    FilesystemProtocol,
    GrepMatch,
    SandboxProtocol,
)
from langchain_agentkit.backends.results import (
    PROBED_TOOLS,
    EditError,
    EditResult,
    FileDownloadResult,
    FileError,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    WriteResult,
)

__all__ = [
    "PROBED_TOOLS",
    "EditError",
    "EditResult",
    "ExecuteResponse",
    "FileDownloadResult",
    "FileError",
    "FileInfo",
    "FileUploadResult",
    "FilesystemProtocol",
    "GrepMatch",
    "ReadBytesResult",
    "ReadResult",
    "SandboxEnvironment",
    "SandboxProtocol",
    "WriteResult",
    "read_tree",
]
