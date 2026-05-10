"""AgentFSBackend — local SQLite-backed virtual filesystem backend.

Implements ``FilesystemProtocol`` on top of the AgentFS Python SDK
(``agentfs-sdk``). AgentFS is a content-addressed virtual filesystem
persisted in a single SQLite/libSQL ``.db`` file, with async file
primitives over ``turso.aio``.

Capability scope: ``FilesystemProtocol`` only.
``SandboxProtocol`` is deliberately not claimed — AgentFS has no
native shell-execution surface (the ``agentfs run`` CLI operates on
a separate ``delta.db`` with the host cwd as the base layer, and
``agentfs exec`` is Linux-only via FUSE / macOS-only via NFS with
undocumented concurrency semantics against a concurrently-open SDK
connection). Users who need shell exec compose ``AgentFSBackend``
with ``OSBackend`` or ``DaytonaBackend``.

Requires ``pip install langchain-agentkit[agentfs]`` (or
``pip install agentfs-sdk`` directly). Importing this module fails
with a clear ``ImportError`` when the SDK isn't installed —
``AgentFSBackend`` isn't re-exported from the generic
``langchain_agentkit.backends`` namespace, so SDK-less users running
``OSBackend`` are unaffected.

Usage::

    from agentfs_sdk import AgentFS, AgentFSOptions
    from langchain_agentkit.backends.agentfs import AgentFSBackend

    agent = await AgentFS.open(AgentFSOptions(path="./agent-state.db"))
    backend = AgentFSBackend(agent)

    result = await backend.read("/notes.md")
    if result.error is None:
        print(result.content)

    await agent.close()
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from typing import TYPE_CHECKING

from agentfs_sdk.errors import ErrnoException  # type: ignore[import-untyped]

from langchain_agentkit.backends.protocol import GrepMatch
from langchain_agentkit.backends.results import (
    EditResult,
    FileDownloadResult,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    WriteResult,
)

if TYPE_CHECKING:
    from agentfs_sdk import AgentFS  # type: ignore[import-untyped]

    from langchain_agentkit.backends.results import FileError


# Map AgentFS POSIX errno codes to the framework's stable FileError vocabulary.
# Codes not listed (ENOTEMPTY, ENOSYS, etc.) fall through to "io_error".
_ERRNO_TO_FILE_ERROR: dict[str, FileError] = {
    "ENOENT": "file_not_found",
    "EISDIR": "is_directory",
    "ENOTDIR": "invalid_path",
    "EPERM": "permission_denied",
    "EINVAL": "invalid_path",
    "EEXIST": "io_error",
}


def _map_errno(exc: ErrnoException) -> FileError:
    return _ERRNO_TO_FILE_ERROR.get(exc.code, "io_error")


class AgentFSBackend:
    """Local SQLite-backed virtual filesystem backend.

    Wraps an already-opened ``AgentFS`` instance. The caller owns the
    lifecycle: ``await AgentFS.open(...)`` before construction,
    ``await agent.close()`` afterward — matching the pre-opened
    constructor shape of ``DaytonaBackend(sandbox)`` and the project's
    "AgentKit is sync; async setup runs separately" convention.

    Args:
        agent: An ``AgentFS`` instance returned by ``await AgentFS.open(...)``.
        workdir: Virtual root inside the ``.db`` for path resolution.
            Defaults to ``"/"`` (the entire virtual filesystem). Pass a
            sub-path (e.g. ``"/workspace"``) to namespace the agent's
            view from app-managed metadata stored at other paths.
    """

    def __init__(self, agent: AgentFS, workdir: str = "/") -> None:
        self._agent = agent
        self._workdir: str = "/" if workdir in ("", "/") else workdir.rstrip("/")

    @property
    def agent(self) -> AgentFS:
        """The underlying AgentFS instance."""
        return self._agent

    @property
    def workdir(self) -> str:
        """The virtual workdir inside the .db."""
        return self._workdir

    # --- Path resolution ---

    def _resolve(self, path: str) -> str:
        """Resolve a virtual path inside the .db, blocking traversal.

        Mirrors ``OSBackend._resolve`` / ``DaytonaBackend._resolve``
        semantics so backends are interchangeable for the LLM. AgentFS
        paths have no host-FS escape risk (the ``.db`` is the world),
        but we still reject ``..`` traversal that escapes the workdir
        — that's part of the protocol contract enforced by the
        conformance suite.
        """
        if not path or path == "/":
            return self._workdir or "/"

        cleaned = path.lstrip("/")
        normalized = posixpath.normpath(cleaned) if cleaned else "."

        # Leading ".." segments after normalization → escape attempt.
        # Detected here rather than via prefix comparison because with
        # workdir="/" the prefix check is degenerate (every absolute
        # path "starts with /").
        if normalized == ".." or normalized.startswith("../"):
            raise PermissionError(f"Path traversal blocked: {path}")
        if normalized == ".":
            return self._workdir or "/"

        if self._workdir == "/":
            return f"/{normalized}"
        return f"{self._workdir}/{normalized}"

    def _strip_workdir(self, abs_path: str) -> str:
        """Convert an absolute .db path back to a virtual path under workdir."""
        if self._workdir == "/":
            return abs_path if abs_path.startswith("/") else f"/{abs_path}"
        if abs_path == self._workdir:
            return "/"
        prefix = self._workdir + "/"
        if abs_path.startswith(prefix):
            return "/" + abs_path[len(prefix) :]
        return abs_path

    # --- FilesystemProtocol: file ops ---

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadResult(error="permission_denied", error_message=str(exc))
        try:
            content = await self._agent.fs.read_file(real_path)
        except ErrnoException as exc:
            return ReadResult(error=_map_errno(exc), error_message=str(exc))
        except UnicodeDecodeError as exc:
            return ReadResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )

        if not isinstance(content, str):
            # Defensive: SDK should return str when encoding defaults to "utf-8";
            # if it ever surfaces bytes here that's a decode failure.
            return ReadResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path}",
            )

        # Slice by line to match OSBackend semantics.
        lines = content.splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        return ReadResult(content="".join(selected))

    async def read_bytes(self, path: str) -> ReadBytesResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadBytesResult(error="permission_denied", error_message=str(exc))
        try:
            content = await self._agent.fs.read_file(real_path, encoding=None)
        except ErrnoException as exc:
            return ReadBytesResult(error=_map_errno(exc), error_message=str(exc))
        if not isinstance(content, (bytes, bytearray)):
            return ReadBytesResult(
                error="io_error",
                error_message=f"Unexpected read_file result type: {type(content).__name__}",
            )
        return ReadBytesResult(content=bytes(content))

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return WriteResult(error="permission_denied", error_message=str(exc))
        try:
            await self._agent.fs.write_file(real_path, content)
        except ErrnoException as exc:
            return WriteResult(error=_map_errno(exc), error_message=str(exc))
        return WriteResult(path=path, bytes_written=len(content))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return EditResult(error="permission_denied", error_message=str(exc))

        try:
            content = await self._agent.fs.read_file(real_path)
        except ErrnoException as exc:
            return EditResult(error=_map_errno(exc), error_message=str(exc))
        except UnicodeDecodeError as exc:
            return EditResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )
        if not isinstance(content, str):
            return EditResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path}",
            )

        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(
                path=path,
                error="old_string_not_found",
                error_message=f"old_string not found in {path}",
            )
        if occurrences > 1 and not replace_all:
            return EditResult(
                path=path,
                occurrences=occurrences,
                error="ambiguous_match",
                error_message=(
                    f"old_string appears {occurrences} times in {path}. "
                    "Pass replace_all=True or extend old_string for uniqueness."
                ),
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        try:
            await self._agent.fs.write_file(real_path, new_content)
        except ErrnoException as exc:
            return EditResult(path=path, error=_map_errno(exc), error_message=str(exc))
        return EditResult(path=path, replacements=occurrences if replace_all else 1)

    async def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Walks the virtual tree via ``readdir`` and applies ``fnmatch``
        for the pattern. The agentfs-sdk has no native path-pattern
        method; the underlying SQLite schema *could* support a
        recursive-CTE optimization via ``agent.get_database()`` for
        large trees, but that's deferred until benchmarks show pain.
        """
        try:
            real_root = self._resolve(path)
        except PermissionError:
            return []

        all_files = await self._walk_files(real_root)

        matches: list[str] = []
        for abs_path in all_files:
            rel = (
                abs_path[len(real_root) :].lstrip("/") if real_root != "/" else abs_path.lstrip("/")
            )
            if fnmatch.fnmatch(rel, pattern):
                matches.append(self._strip_workdir(abs_path))
        return sorted(matches)

    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]:
        """Search file contents for a regex pattern."""
        try:
            real_root = self._resolve(path or "/")
        except PermissionError:
            return []

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            return []

        matches: list[GrepMatch] = []
        all_files = await self._walk_files(real_root)
        for abs_path in all_files:
            if glob and not fnmatch.fnmatch(posixpath.basename(abs_path), glob):
                continue
            try:
                content = await self._agent.fs.read_file(abs_path)
            except ErrnoException:
                continue
            except UnicodeDecodeError:
                continue
            if not isinstance(content, str):
                continue
            virtual_path = self._strip_workdir(abs_path)
            for i, line in enumerate(content.splitlines(keepends=True), start=1):
                if regex.search(line):
                    matches.append(GrepMatch(path=virtual_path, line=i, text=line))
        return matches

    # --- FilesystemProtocol: bulk transfer (host-side) ---
    #
    # AgentFS has no native multipart endpoint — the .db is local and
    # ``write_file`` / ``read_file`` are already cheap (one SQLite
    # transaction each). Bulk transfer is implemented as a per-file
    # loop. The capability claim is honest: callers can transfer many
    # files in one call; the protocol describes capability, not
    # bulk-network optimization.

    async def upload(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]:
        results: list[FileUploadResult] = []
        for path, content in files:
            wr = await self.write(path, content)
            if wr.error is not None:
                results.append(
                    FileUploadResult(path=path, error=wr.error, error_message=wr.error_message)
                )
            else:
                results.append(FileUploadResult(path=path, bytes_written=wr.bytes_written))
        return results

    async def download(self, paths: list[str]) -> list[FileDownloadResult]:
        results: list[FileDownloadResult] = []
        for path in paths:
            rb = await self.read_bytes(path)
            if rb.error is not None:
                results.append(
                    FileDownloadResult(path=path, error=rb.error, error_message=rb.error_message)
                )
            else:
                results.append(FileDownloadResult(path=path, content=rb.content))
        return results

    # --- Internal helpers ---

    async def _walk_files(self, root: str) -> list[str]:
        """Recursively collect all regular-file paths under ``root``.

        Uses ``fs.readdir`` + ``fs.stat`` to classify entries.
        Symlinks are skipped (AgentFS schema declares them but ops
        explicitly raise ``ENOSYS`` on symlink paths today).
        """
        try:
            entries = await self._agent.fs.readdir(root)
        except ErrnoException:
            return []

        files: list[str] = []
        for name in entries:
            child = "/" + name if root == "/" else root + "/" + name
            try:
                st = await self._agent.fs.stat(child)
            except ErrnoException:
                continue
            if st.is_directory():
                files.extend(await self._walk_files(child))
            elif st.is_file():
                files.append(child)
        return files
