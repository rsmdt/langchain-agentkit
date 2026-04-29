"""Backend implementations for agent environments.

Two backends are provided:

- ``OSBackend`` — native local filesystem via Python stdlib.
- ``DaytonaBackend`` — Daytona cloud sandbox via shell commands.

Both implement all three capability protocols: ``BackendProtocol``,
``SandboxBackend``, and ``FileTransferBackend``.

Usage::

    from langchain_agentkit.backends import (
        BackendProtocol, SandboxBackend, FileTransferBackend,
        OSBackend, DaytonaBackend,
    )
"""

from langchain_agentkit.backends.daytona import DaytonaBackend
from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.backends.protocol import (
    BackendProtocol,
    ExecuteResponse,
    FileInfo,
    FileTransferBackend,
    GrepMatch,
    SandboxBackend,
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
    "BackendProtocol",
    "DaytonaBackend",
    "EditError",
    "EditResult",
    "ExecuteResponse",
    "FileDownloadResult",
    "FileError",
    "FileInfo",
    "FileTransferBackend",
    "FileUploadResult",
    "GrepMatch",
    "OSBackend",
    "ReadBytesResult",
    "ReadResult",
    "SandboxBackend",
    "SandboxEnvironment",
    "WriteResult",
]
