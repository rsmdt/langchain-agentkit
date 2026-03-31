"""Backend protocol — the 6-method interface for agent environments.

Mirrors the Claude Code tool surface: Read, Write, Edit, Glob, Grep, Bash.

Any environment (local OS, Docker, Daytona, Modal) implements this protocol.
With ``BaseSandbox``, a new provider only needs to implement ``execute()``.

Usage::

    from langchain_agentkit.backends import BackendProtocol, BaseSandbox

    # Full implementation
    class MyBackend:
        def read(self, path, offset=0, limit=2000) -> str: ...
        def write(self, path, content) -> WriteResult: ...
        def edit(self, path, old_string, new_string, replace_all=False) -> EditResult: ...
        def glob(self, pattern, path="/") -> list[str]: ...
        def grep(self, pattern, path=None, glob=None, ignore_case=False) -> list[GrepMatch]: ...
        def execute(self, command, timeout=None, workdir=None) -> ExecuteResponse: ...

    # Minimal implementation (sandbox with shell access)
    class MySandbox(BaseSandbox):
        def execute(self, command, timeout=None, workdir=None) -> ExecuteResponse: ...
        # read, write, edit, glob, grep all inherited via shell commands
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


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendProtocol(Protocol):
    """The 6-method interface for agent environments.

    Mirrors the Claude Code tool surface:
    - ``read`` → Read
    - ``write`` → Write
    - ``edit`` → Edit
    - ``glob`` → Glob
    - ``grep`` → Grep
    - ``execute`` → Bash
    """

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str: ...
    def write(self, path: str, content: str | bytes) -> WriteResult: ...
    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult: ...
    def glob(self, pattern: str, path: str = "/") -> list[str]: ...
    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]: ...
    def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse: ...
