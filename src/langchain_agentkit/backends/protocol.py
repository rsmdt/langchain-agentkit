"""Backend protocol — the interface for agent environments.

Pure data-access layer. Returns raw content — presentation concerns
(line numbers, formatting) belong in the tool layer above.

All methods are async to support both local and remote backends.

Two implementations are provided:

- ``OSBackend`` — native local filesystem via Python stdlib.
- ``DaytonaBackend`` — Daytona cloud sandbox via shell commands.

Usage::

    from langchain_agentkit.backends import BackendProtocol

    class MyBackend:
        async def read(self, path, offset=0, limit=2000) -> str: ...
        async def read_bytes(self, path) -> bytes: ...
        async def write(self, path, content) -> WriteResult: ...
        async def edit(self, path, old_string, new_string, replace_all=False) -> EditResult: ...
        async def glob(self, pattern, path="/") -> list[str]: ...
        async def grep(self, pattern, path=None, glob=None,
                       ignore_case=False) -> list[GrepMatch]: ...
        async def execute(self, command, timeout=None, workdir=None) -> ExecuteResponse: ...
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FileInfo(TypedDict):
    path: str
    size: int
    is_dir: bool


class WriteResult(TypedDict):
    path: str
    bytes_written: int


class EditResult(TypedDict):
    path: str
    replacements: int


class GrepMatch(TypedDict):
    path: str
    line: int
    text: str


class ExecuteResponse(TypedDict):
    output: str
    exit_code: int
    truncated: bool
    stderr: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendProtocol(Protocol):
    """The 7-method async interface for agent environments.

    All methods are coroutines. Covers the standard tool surface:
    - ``read`` → Read
    - ``write`` → Write
    - ``edit`` → Edit
    - ``glob`` → Glob
    - ``grep`` → Grep
    - ``execute`` → Bash
    """

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> str: ...
    async def read_bytes(self, path: str) -> bytes: ...
    async def write(self, path: str, content: str | bytes) -> WriteResult: ...
    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult: ...
    async def glob(self, pattern: str, path: str = "/") -> list[str]: ...
    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]: ...
    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse: ...
