"""Backend protocol and OS filesystem implementation for agent file operations.

``BackendProtocol`` defines file operations (read, write, edit, glob, grep, ls).
``SandboxProtocol`` extends it with shell execution.

Implementations:
- ``OSBackend`` — real OS filesystem with path traversal prevention
"""

from __future__ import annotations

import fnmatch
import os
import re
from typing import Protocol, TypedDict, runtime_checkable

# --- Data types ---


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


# --- Protocols ---


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol for file operations."""

    def ls(self, path: str) -> list[FileInfo]: ...
    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str: ...
    def write(self, path: str, content: str | bytes) -> WriteResult: ...
    def edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False,
    ) -> EditResult: ...
    def glob(self, pattern: str, path: str = "/") -> list[str]: ...
    def grep(
        self, pattern: str, path: str | None = None,
        glob: str | None = None, ignore_case: bool = False,
    ) -> list[GrepMatch]: ...
    def exists(self, path: str) -> bool: ...
    def delete(self, path: str) -> None: ...


@runtime_checkable
class SandboxProtocol(BackendProtocol, Protocol):
    """Protocol extending BackendProtocol with shell execution."""

    def execute(
        self, command: str, timeout: int | None = None, workdir: str | None = None,
    ) -> ExecuteResponse: ...


# --- OSBackend ---


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
        """Resolve path relative to root, blocking traversal."""
        cleaned = path.lstrip("/")
        resolved = os.path.realpath(os.path.join(self._root, cleaned))
        if not resolved.startswith(self._root):
            raise PermissionError(f"Path traversal blocked: {path}")
        return resolved

    def ls(self, path: str) -> list[FileInfo]:
        real_path = self._resolve(path)
        entries: list[FileInfo] = []
        if os.path.isdir(real_path):
            for name in sorted(os.listdir(real_path)):
                full = os.path.join(real_path, name)
                rel = os.path.join(path.rstrip("/"), name)
                entries.append(FileInfo(
                    path=rel,
                    size=os.path.getsize(full) if os.path.isfile(full) else 0,
                    is_dir=os.path.isdir(full),
                ))
        return entries

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        real_path = self._resolve(path)
        with open(real_path) as f:
            lines = f.readlines()
        selected = lines[offset : offset + limit]
        return "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))

    def write(self, path: str, content: str | bytes) -> WriteResult:
        real_path = self._resolve(path)
        os.makedirs(os.path.dirname(real_path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(real_path, mode) as f:
            f.write(content)
        return WriteResult(path=path, bytes_written=len(content))

    def edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False,
    ) -> EditResult:
        real_path = self._resolve(path)
        with open(real_path) as f:
            content = f.read()
        if replace_all:
            count = content.count(old_string)
            new_content = content.replace(old_string, new_string)
        else:
            count = 1 if old_string in content else 0
            new_content = content.replace(old_string, new_string, 1)
        with open(real_path, "w") as f:
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
        self, pattern: str, path: str | None = None,
        glob: str | None = None, ignore_case: bool = False,
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
                    with open(full) as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(GrepMatch(path=rel, line=i, text=line.rstrip()))
                except (OSError, UnicodeDecodeError):
                    continue
        return matches

    def exists(self, path: str) -> bool:
        real_path = self._resolve(path)
        return os.path.exists(real_path)

    def delete(self, path: str) -> None:
        real_path = self._resolve(path)
        if os.path.isfile(real_path):
            os.remove(real_path)
        elif os.path.isdir(real_path):
            import shutil

            shutil.rmtree(real_path)
