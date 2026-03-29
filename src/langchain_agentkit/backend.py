"""Backend protocol and implementations for agent file/execution operations.

``BackendProtocol`` defines file operations (read, write, edit, glob, grep, ls).
``SandboxProtocol`` extends it with shell execution.

Implementations:
- ``MemoryBackend`` — wraps VirtualFilesystem (in-memory, testing/ephemeral)
- ``LocalBackend`` — real filesystem with path traversal prevention
- ``CompositeBackend`` — routes by path prefix to different backends
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable


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
    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult: ...
    def glob(self, pattern: str, path: str = "/") -> list[str]: ...
    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch]: ...
    def exists(self, path: str) -> bool: ...
    def delete(self, path: str) -> None: ...


@runtime_checkable
class SandboxProtocol(BackendProtocol, Protocol):
    """Protocol extending BackendProtocol with shell execution."""

    def execute(self, command: str, timeout: int | None = None, workdir: str | None = None) -> ExecuteResponse: ...


# --- MemoryBackend ---


class MemoryBackend:
    """In-memory backend wrapping VirtualFilesystem.

    Suitable for testing and ephemeral agents.
    """

    def __init__(self) -> None:
        from langchain_agentkit.vfs import VirtualFilesystem

        self._vfs = VirtualFilesystem()

    def ls(self, path: str) -> list[FileInfo]:
        entries = self._vfs.list_directory(path)
        normalized = self._vfs.normalize_path(path)
        if not normalized.endswith("/"):
            normalized += "/"
        result: list[FileInfo] = []
        for entry in entries:
            name = str(entry)
            is_dir = name.endswith("/")
            full_path = normalized + name.rstrip("/")
            result.append(FileInfo(path=full_path, size=0, is_dir=is_dir))
        return result

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        content = self._vfs.read(path)
        if content is None:
            raise FileNotFoundError(f"File not found: {path}")
        lines = content.splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        return "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(selected))

    def write(self, path: str, content: str | bytes) -> WriteResult:
        text = content if isinstance(content, str) else content.decode("utf-8")
        self._vfs.write(path, text)
        return WriteResult(path=path, bytes_written=len(content))

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        count = self._vfs.edit(path, old_string, new_string, replace_all=replace_all)
        return EditResult(path=path, replacements=count)

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        # VFS glob takes a full pattern, so combine path + pattern
        if path and path != "/":
            full_pattern = path.rstrip("/") + "/" + pattern
        else:
            full_pattern = pattern if pattern.startswith("/") else "/" + pattern
        return self._vfs.glob(full_pattern)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch]:
        results = self._vfs.grep(pattern, path=path, glob_filter=glob)
        matches: list[GrepMatch] = []
        for r in results:
            matches.append(GrepMatch(
                path=str(r.get("path", "")),
                line=int(r.get("line", 0)),
                text=str(r.get("text", "")),
            ))
        return matches

    def exists(self, path: str) -> bool:
        return self._vfs.exists(path)

    def delete(self, path: str) -> None:
        self._vfs.delete(path)


# --- LocalBackend ---


class LocalBackend:
    """Real filesystem backend with path traversal prevention.

    All paths are resolved relative to ``root_dir``. Any path that
    escapes the root raises ``PermissionError``.

    Args:
        root_dir: The root directory for all file operations.
    """

    def __init__(self, root_dir: str) -> None:
        self._root = os.path.realpath(root_dir)

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

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
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

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch]:
        search_path = self._resolve(path or "/")
        matches: list[GrepMatch] = []
        regex = re.compile(pattern)

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


# --- CompositeBackend ---


class CompositeBackend:
    """Routes operations to different backends by path prefix.

    Uses longest-prefix matching. Unmatched paths go to the default backend.

    Args:
        default: Backend for paths that don't match any route.
        routes: Dict mapping path prefixes to backends.
    """

    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol] | None = None,
    ) -> None:
        self._default = default
        # Sort by prefix length (longest first) for longest-prefix matching
        self._routes = sorted(
            (routes or {}).items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

    def _route(self, path: str) -> tuple[Any, str]:
        """Find the backend for a path and strip the prefix."""
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                return backend, path[len(prefix):]
        return self._default, path

    def ls(self, path: str) -> list[FileInfo]:
        backend, resolved = self._route(path)
        return backend.ls(resolved)

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        backend, resolved = self._route(path)
        return backend.read(resolved, offset, limit)

    def write(self, path: str, content: str | bytes) -> WriteResult:
        backend, resolved = self._route(path)
        return backend.write(resolved, content)

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        backend, resolved = self._route(path)
        return backend.edit(resolved, old_string, new_string, replace_all)

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        backend, resolved = self._route(path)
        return backend.glob(pattern, resolved)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch]:
        backend, resolved = self._route(path or "/")
        return backend.grep(pattern, resolved or None, glob)

    def exists(self, path: str) -> bool:
        backend, resolved = self._route(path)
        return backend.exists(resolved)

    def delete(self, path: str) -> None:
        backend, resolved = self._route(path)
        backend.delete(resolved)
