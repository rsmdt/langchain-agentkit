"""Backend protocols — capability-tiered interfaces for agent environments.

Pure data-access layer. Returns raw content; presentation concerns
(line numbers, formatting) belong in the tool layer above.

All methods are async. Three structural ``Protocol`` tiers:

- :class:`BackendProtocol` — file operations. Required for every backend.
- :class:`SandboxBackend` — adds shell ``execute``. Backends that
  satisfy this signal that the bash tool may be registered.
- :class:`FileTransferBackend` — adds bulk binary ``upload_files`` and
  ``download_files``. Used by host-side seeding/extraction; **not**
  exposed as an LLM tool.

Capability gating is structural: ``isinstance(backend, SandboxBackend)``
returns ``True`` iff the methods are present. Tool factories use that
check to decide which tools to register; backends that lack a capability
simply produce an agent without the corresponding tool.

Expected, LLM-actionable failures (file not found, ambiguous edit, etc.)
return result dataclasses with an ``error`` code from a stable ``Literal``
set. Unexpected failures (network down, SDK bug) raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from langchain_agentkit.backends.results import (
        EditResult,
        FileDownloadResult,
        FileUploadResult,
        ReadBytesResult,
        ReadResult,
        SandboxEnvironment,
        WriteResult,
    )

# ---------------------------------------------------------------------------
# Auxiliary data types (non-result shapes that don't carry error codes)
# ---------------------------------------------------------------------------


class FileInfo(TypedDict):
    path: str
    size: int
    is_dir: bool


class GrepMatch(TypedDict):
    path: str
    line: int
    text: str


class ExecuteResponse(TypedDict, total=False):
    """Result of a backend shell execution.

    Keys:
        output: Captured stdout tail. When ``output_path`` is set, this
            holds the trailing window of the full output that fit in the
            in-memory buffer; the complete stream is on disk.
        exit_code: Process exit code (``-1`` on timeout or spawn failure).
        truncated: ``True`` when output exceeded the buffer cap or the
            command timed out.
        stderr: Captured stderr tail (same buffer semantics as ``output``).
        output_path: Path to a temp file containing the FULL combined
            stdout + stderr when the buffer overflowed. ``None`` when the
            output fit in memory.
        lines_dropped: Number of stdout + stderr lines dropped from the
            tail window. Zero when ``truncated`` is ``False``.
        bytes_dropped: Number of bytes dropped from the head of stdout +
            stderr before the tail window begins. Zero when ``truncated``
            is ``False``.
    """

    output: str
    exit_code: int
    truncated: bool
    stderr: str
    output_path: str | None
    lines_dropped: int
    bytes_dropped: int


# ---------------------------------------------------------------------------
# Capability-tier protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendProtocol(Protocol):
    """Base file capability — required for every backend.

    Methods that may fail in expected, LLM-actionable ways return result
    dataclasses with optional ``error`` codes. ``glob`` and ``grep`` have
    no per-call error semantics worth a wrapper and return lists directly.
    """

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult: ...

    async def read_bytes(self, path: str) -> ReadBytesResult: ...

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


@runtime_checkable
class SandboxBackend(BackendProtocol, Protocol):
    """Adds shell execution and environment introspection.

    Tool factories register the bash tool only when ``isinstance(backend,
    SandboxBackend)`` is true. The ``FilesystemExtension`` calls
    ``environment()`` during async setup and surfaces the result to the
    LLM as the ``<env>`` block of the system prompt — the LLM uses it to
    pick correct shell-flag dialect and reach for available tools.
    """

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse: ...

    async def environment(self) -> SandboxEnvironment: ...


@runtime_checkable
class FileTransferBackend(BackendProtocol, Protocol):
    """Adds bulk binary I/O. Used by host-side seeding and extraction.

    Not exposed as an LLM tool. The ``seed_directory`` helper requires
    this capability and replaces ad-hoc per-file write loops.
    """

    async def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]: ...

    async def download_files(self, paths: list[str]) -> list[FileDownloadResult]: ...
