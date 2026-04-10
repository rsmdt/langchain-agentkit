"""OSBackend — local filesystem + subprocess backend.

Implements the full ``BackendProtocol`` using the real OS filesystem
with path traversal prevention and ``subprocess.run`` for execute.

Usage::

    from langchain_agentkit.backends import OSBackend

    backend = OSBackend("/path/to/workspace")
    content = backend.read("/app/main.py")  # raw text, no line numbers
    result = backend.execute("python3 -m pytest")
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import re
import subprocess
from typing import Any

from langchain_agentkit.backends.protocol import (
    EditResult,
    ExecuteResponse,
    GrepMatch,
    WriteResult,
)


class OSBackend:
    """Real OS filesystem backend with path traversal prevention.

    All paths are resolved relative to ``root``. Any path that
    escapes the root raises ``PermissionError``.

    Args:
        root: The root directory for all file operations.
    """

    def __init__(self, root: str) -> None:
        self._root = os.path.realpath(root)

    def _resolve(self, path: str) -> str:
        """Resolve path relative to root, blocking traversal.

        Accepts three forms:
        - Root-relative: ``/workspace/file.txt`` (leading ``/`` stripped)
        - Absolute inside root: ``/tmp/sandbox123/workspace/file.txt``
        - Traversal / outside root: blocked with ``PermissionError``
        """
        root_with_sep = self._root.rstrip(os.sep) + os.sep

        # Check if the raw path already resolves inside root (handles
        # absolute paths the LLM may construct from the prompt).
        real_path = os.path.realpath(path)
        if real_path == self._root or real_path.startswith(root_with_sep):
            return real_path

        # Treat as root-relative: strip leading "/" and join with root.
        cleaned = path.lstrip("/")
        resolved = os.path.realpath(os.path.join(self._root, cleaned))
        if resolved != self._root and not resolved.startswith(root_with_sep):
            raise PermissionError(f"Path traversal blocked: {path}")
        return resolved

    # --- BackendProtocol methods ---

    def read_bytes(self, path: str) -> bytes:
        real_path = self._resolve(path)
        with open(real_path, "rb") as f:
            return f.read()

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read raw text content with offset/limit support."""
        real_path = self._resolve(path)
        with open(real_path, encoding="utf-8") as f:
            selected = list(itertools.islice(itertools.islice(f, offset, None), limit))
        return "".join(selected)

    def write(self, path: str, content: str | bytes) -> WriteResult:
        real_path = self._resolve(path)
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(real_path, mode, **kwargs) as f:  # type: ignore[call-overload]
            f.write(content)
        return WriteResult(path=path, bytes_written=len(content))

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        real_path = self._resolve(path)
        with open(real_path, encoding="utf-8") as f:
            content = f.read()
        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(path=path, replacements=0)
        if replace_all:
            new_content = content.replace(old_string, new_string)
            count = occurrences
        else:
            if occurrences > 1:
                raise ValueError(
                    f"Ambiguous edit: old_string appears {occurrences} times in "
                    f"{path}. Use replace_all=True or provide more context to "
                    f"make the match unique."
                )
            new_content = content.replace(old_string, new_string, 1)
            count = 1
        with open(real_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return EditResult(path=path, replacements=count)

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        real_path = self._resolve(path)
        import glob as glob_mod

        full_pattern = os.path.join(real_path, pattern)
        matches = glob_mod.glob(full_pattern, recursive=True)
        result: list[str] = []
        for m in sorted(matches):
            rel = os.path.relpath(m, self._root)
            result.append("/" + rel)
        return result

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]:
        search_path = self._resolve(path or "/")
        matches: list[GrepMatch] = []
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)

        for root, _dirs, files in os.walk(search_path):
            for fname in sorted(files):
                if glob and not fnmatch.fnmatch(fname, glob):
                    continue
                full = os.path.join(root, fname)
                rel = "/" + os.path.relpath(full, self._root)
                try:
                    with open(full, encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(GrepMatch(path=rel, line=i, text=line))
                except (OSError, UnicodeDecodeError):
                    continue
        return matches

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command via subprocess."""
        cwd = self._resolve(workdir) if workdir else self._root
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return ExecuteResponse(
                output=result.stdout,
                exit_code=result.returncode,
                truncated=False,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecuteResponse(
                output=str(exc),
                exit_code=-1,
                truncated=True,
                stderr="",
            )

    # --- Convenience methods (not part of BackendProtocol) ---

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
            import shutil

            shutil.rmtree(real_path)
