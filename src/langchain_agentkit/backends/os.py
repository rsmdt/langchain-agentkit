"""OSBackend — local filesystem + subprocess backend.

Implements ``FilesystemProtocol`` and ``SandboxProtocol`` using
the real OS filesystem with path traversal
prevention and ``asyncio.create_subprocess_shell`` for execute.

Usage::

    from langchain_agentkit.backends.os import OSBackend

    backend = OSBackend("/path/to/workspace")
    result = await backend.read("/app/main.py")
    if result.error is None:
        print(result.content)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import re
import shutil
import signal
from typing import TYPE_CHECKING, Any

from langchain_agentkit.backends.edits import empty_old_string_result
from langchain_agentkit.backends.execution import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    BoundedCapture,
    drain_stream_into,
)
from langchain_agentkit.backends.protocol import (
    ExecuteResponse,
    GrepMatch,
)
from langchain_agentkit.backends.results import (
    PROBED_TOOLS,
    EditResult,
    FileDownloadResult,
    FileUploadResult,
    GlobResult,
    GrepResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    WriteResult,
)
from langchain_agentkit.backends.search import compile_glob_pattern, summarize_search_errors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_POSIX = os.name == "posix"


class OSBackend:
    """Real OS filesystem backend with path traversal prevention.

    All paths are resolved relative to ``root``. Any path that escapes
    the root is reported as ``permission_denied``.

    Args:
        root: The root directory for all file operations.
    """

    def __init__(
        self,
        root: str,
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    ) -> None:
        self._root = os.path.realpath(root)
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._env_cache: SandboxEnvironment | None = None

    def _resolve(self, path: str) -> str:
        """Resolve path relative to root, blocking traversal.

        Raises ``PermissionError`` for paths that escape the root.
        Backend methods catch this and convert to a ``permission_denied``
        error code; ``execute`` workdir resolution lets it propagate.
        """
        root_with_sep = self._root.rstrip(os.sep) + os.sep
        real_path = os.path.realpath(path)
        if real_path == self._root or real_path.startswith(root_with_sep):
            return real_path
        cleaned = path.lstrip("/")
        resolved = os.path.realpath(os.path.join(self._root, cleaned))
        if resolved != self._root and not resolved.startswith(root_with_sep):
            raise PermissionError(f"Path traversal blocked: {path}")
        return resolved

    # --- FilesystemProtocol ---

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadResult(error="permission_denied", error_message=str(exc))
        start = max(0, offset)
        end = start + max(0, limit)
        try:
            selected: list[str] = []
            total_lines = 0
            with open(real_path, encoding="utf-8") as f:
                for line in f:
                    if start <= total_lines < end:
                        selected.append(line)
                    total_lines += 1
            return ReadResult(content="".join(selected), total_lines=total_lines)
        except FileNotFoundError:
            return ReadResult(error="file_not_found", error_message=f"File not found: {path}")
        except IsADirectoryError:
            return ReadResult(error="is_directory", error_message=f"Path is a directory: {path}")
        except UnicodeDecodeError as exc:
            return ReadResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )
        except OSError as exc:
            return ReadResult(error="io_error", error_message=str(exc))

    async def read_bytes(self, path: str) -> ReadBytesResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadBytesResult(error="permission_denied", error_message=str(exc))
        try:
            with open(real_path, "rb") as f:
                return ReadBytesResult(content=f.read())
        except FileNotFoundError:
            return ReadBytesResult(error="file_not_found", error_message=f"File not found: {path}")
        except IsADirectoryError:
            return ReadBytesResult(
                error="is_directory", error_message=f"Path is a directory: {path}"
            )
        except OSError as exc:
            return ReadBytesResult(error="io_error", error_message=str(exc))

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return WriteResult(error="permission_denied", error_message=str(exc))
        try:
            os.makedirs(os.path.dirname(real_path), exist_ok=True)
            mode = "wb" if isinstance(content, bytes) else "w"
            kwargs: dict[str, Any] = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
            with open(real_path, mode, **kwargs) as f:
                f.write(content)
            return WriteResult(path=path, bytes_written=len(content))
        except IsADirectoryError:
            return WriteResult(error="is_directory", error_message=f"Path is a directory: {path}")
        except OSError as exc:
            return WriteResult(error="io_error", error_message=str(exc))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if not old_string:
            return empty_old_string_result(path)
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return EditResult(error="permission_denied", error_message=str(exc))
        try:
            with open(real_path, encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return EditResult(error="file_not_found", error_message=f"File not found: {path}")
        except IsADirectoryError:
            return EditResult(error="is_directory", error_message=f"Path is a directory: {path}")
        except UnicodeDecodeError as exc:
            return EditResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )
        except OSError as exc:
            return EditResult(error="io_error", error_message=str(exc))

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
        count = occurrences if replace_all else 1
        try:
            with open(real_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as exc:
            return EditResult(path=path, error="io_error", error_message=str(exc))
        return EditResult(path=path, replacements=count)

    async def glob(self, pattern: str, path: str = "/", max_results: int = 1000) -> GlobResult:
        try:
            search_root = self._search_root(path)
        except (PermissionError, NotADirectoryError) as exc:
            return GlobResult(error_message=str(exc))
        matches = compile_glob_pattern(pattern)
        errors: list[str] = []
        paths: list[str] = []
        for real_path in self._walk_files(search_root, errors):
            if not matches(os.path.relpath(real_path, search_root).replace(os.sep, "/")):
                continue
            if len(paths) == max_results:
                return GlobResult(
                    paths=paths, truncated=True, error_message=summarize_search_errors(errors)
                )
            paths.append(self._virtual(real_path))
        return GlobResult(paths=paths, error_message=summarize_search_errors(errors))

    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
        max_matches: int = 1000,
    ) -> GrepResult:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return GrepResult(error_message=f"Invalid regex {pattern!r}: {exc}")
        try:
            search_root = self._search_root(path or "/")
        except (PermissionError, NotADirectoryError) as exc:
            return GrepResult(error_message=str(exc))
        include: Callable[[str], bool] | None = compile_glob_pattern(glob) if glob else None
        errors: list[str] = []
        matches: list[GrepMatch] = []
        for real_path in self._walk_files(search_root, errors):
            relative = os.path.relpath(real_path, search_root).replace(os.sep, "/")
            if include is not None and not include(relative):
                continue
            virtual_path = self._virtual(real_path)
            try:
                with open(real_path, encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        if not regex.search(line):
                            continue
                        if len(matches) == max_matches:
                            return GrepResult(
                                matches=matches,
                                truncated=True,
                                error_message=summarize_search_errors(errors),
                            )
                        matches.append(GrepMatch(path=virtual_path, line=line_number, text=line))
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                errors.append(f"{exc.strerror or exc}: {virtual_path}")
        return GrepResult(matches=matches, error_message=summarize_search_errors(errors))

    def _search_root(self, path: str) -> str:
        search_root = self._resolve(path)
        if not os.path.isdir(search_root):
            raise NotADirectoryError(f"Directory not found: {path}")
        return search_root

    def _virtual(self, real_path: str) -> str:
        relative = os.path.relpath(real_path, self._root).replace(os.sep, "/")
        return "/" if relative == "." else "/" + relative

    def _walk_files(self, directory: str, errors: list[str]) -> Iterator[str]:
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
        except OSError as exc:
            errors.append(f"{exc.strerror or exc}: {self._virtual(directory)}")
            return
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from self._walk_files(entry.path, errors)
                elif entry.is_file() and self._stays_inside_root(entry):
                    yield entry.path
            except OSError as exc:
                errors.append(f"{exc.strerror or exc}: {self._virtual(entry.path)}")

    def _stays_inside_root(self, entry: os.DirEntry[str]) -> bool:
        if not entry.is_symlink():
            return True
        target = os.path.realpath(entry.path)
        return target.startswith(self._root.rstrip(os.sep) + os.sep)

    # --- SandboxProtocol ---

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command, streaming into a bounded capture.

        Output is streamed into :class:`BoundedCapture` — stdout and stderr
        each keep a rolling tail of the last ``max_output_bytes`` /
        ``max_output_lines`` while the full interleaved transcript is
        mirrored to a temp file. The temp file survives only when output
        overflows the tail; happy-path runs leave no litter.

        On timeout the process tree is killed and whatever streamed so far
        is returned with ``truncated=True``. On any other failure the
        capture is abandoned (spill file deleted) and the exception
        propagates.
        """
        cwd = self._resolve(workdir) if workdir else self._root
        capture = BoundedCapture(
            stdout_max_bytes=self._max_output_bytes,
            stdout_max_lines=self._max_output_lines,
            stderr_max_bytes=self._max_output_bytes,
            stderr_max_lines=self._max_output_lines,
        )
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=_POSIX,
            )
            pump = asyncio.gather(
                drain_stream_into(proc.stdout, capture.feed_stdout),
                drain_stream_into(proc.stderr, capture.feed_stderr),
                proc.wait(),
            )
            try:
                await asyncio.wait_for(pump, timeout=timeout)
            except TimeoutError:
                _kill_process_tree(proc)
                with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    await asyncio.wait_for(pump, timeout=2.0)
                stdout_res, stderr_res, spill_path = capture.finalize()
                return ExecuteResponse(
                    output=stdout_res.tail.decode("utf-8", errors="replace"),
                    stderr=stderr_res.tail.decode("utf-8", errors="replace"),
                    exit_code=-1,
                    truncated=True,
                    output_path=str(spill_path) if spill_path is not None else None,
                    lines_dropped=stdout_res.lines_dropped + stderr_res.lines_dropped,
                    bytes_dropped=stdout_res.bytes_dropped + stderr_res.bytes_dropped,
                )
        except BaseException:
            capture.abandon()
            if proc is not None and proc.returncode is None:
                _kill_process_tree(proc)
            raise

        stdout_res, stderr_res, spill_path = capture.finalize()
        truncated = spill_path is not None
        return ExecuteResponse(
            output=stdout_res.tail.decode("utf-8", errors="replace"),
            stderr=stderr_res.tail.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
            truncated=truncated,
            output_path=str(spill_path) if spill_path is not None else None,
            lines_dropped=stdout_res.lines_dropped + stderr_res.lines_dropped,
            bytes_dropped=stdout_res.bytes_dropped + stderr_res.bytes_dropped,
        )

    async def environment(self) -> SandboxEnvironment:
        """Return the shell-environment snapshot for this backend.

        Cheap on OSBackend — backend equals the agent process, so we can
        read ``platform.*`` and ``shutil.which`` directly. Result is
        cached for the backend's lifetime.
        """
        if self._env_cache is None:
            self._env_cache = SandboxEnvironment(
                os=f"{platform.system()} {platform.release()} {platform.machine()}",
                shell=os.environ.get("SHELL", "/bin/sh"),
                cwd=self._root,
                available_tools=frozenset(t for t in PROBED_TOOLS if shutil.which(t)),
            )
        return self._env_cache

    # --- FilesystemProtocol: bulk transfer (host-side) ---

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

    # --- Convenience methods (not part of any protocol) ---

    def ls(self, path: str) -> list[dict[str, Any]]:
        """List directory contents. Not part of the protocol — use Glob or Bash."""
        real_path = self._resolve(path)
        entries: list[dict[str, Any]] = []
        if os.path.isdir(real_path):
            for name in sorted(os.listdir(real_path)):
                full = os.path.join(real_path, name)
                rel = os.path.join(path.rstrip("/"), name)
                entries.append(
                    {
                        "path": rel,
                        "size": os.path.getsize(full) if os.path.isfile(full) else 0,
                        "is_dir": os.path.isdir(full),
                    }
                )
        return entries

    def exists(self, path: str) -> bool:
        """Check if a path exists. Not part of the protocol — use Glob or Read."""
        real_path = self._resolve(path)
        return os.path.exists(real_path)

    def delete(self, path: str) -> None:
        """Delete a file or directory. Not part of the protocol — use Bash."""
        real_path = self._resolve(path)
        if os.path.isfile(real_path):
            os.remove(real_path)
        elif os.path.isdir(real_path):
            shutil.rmtree(real_path)


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if _POSIX:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
