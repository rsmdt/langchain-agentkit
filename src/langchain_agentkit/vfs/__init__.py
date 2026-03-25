"""In-memory virtual filesystem for agent tool access.

Stores files as ``{path: content}`` dicts with POSIX-style paths.
All paths are normalized to absolute form (leading ``/``).

Usage::

    from langchain_agentkit.vfs import VirtualFilesystem

    vfs = VirtualFilesystem()
    vfs.write("/skills/web-research/SKILL.md", "# Web Research\\n...")
    content = vfs.read("/skills/web-research/SKILL.md")
"""

from __future__ import annotations

import fnmatch
import re


def _glob_to_regex(pattern: str) -> str:
    """Convert a glob pattern to a regex string.

    Handles ``**`` (match any path segments), ``*`` (match within segment),
    and ``?`` (match single char).
    """
    parts = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # ** matches any number of path segments
                parts.append(".*")
                i += 2
                # Skip trailing /
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            else:
                # * matches anything except /
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        elif c in ".+^${}()|[]":
            parts.append("\\" + c)
            i += 1
        else:
            parts.append(c)
            i += 1
    return "^" + "".join(parts) + "$"


class VirtualFilesystem:
    """In-memory filesystem with POSIX-style paths.

    Files are stored as plain strings keyed by absolute paths.
    Directories are implicit — they exist when files exist beneath them.

    Thread-safety: not thread-safe. Use one instance per agent.
    """

    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize a path to absolute POSIX form.

        - Resolves ``..`` and ``.`` components
        - Ensures leading ``/``
        - Strips trailing ``/``
        """
        if not path.startswith("/"):
            path = "/" + path
        # Resolve . and .. manually since PurePosixPath doesn't resolve ..
        parts: list[str] = []
        for part in path.split("/"):
            if part == "" or part == ".":
                continue
            elif part == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(part)
        return "/" + "/".join(parts) if parts else "/"

    def write(self, path: str, content: str) -> None:
        """Write content to a file, creating or overwriting."""
        self._files[self.normalize_path(path)] = content

    def read(self, path: str) -> str | None:
        """Read file content. Returns ``None`` if file does not exist."""
        return self._files.get(self.normalize_path(path))

    def exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        return self.normalize_path(path) in self._files

    def delete(self, path: str) -> bool:
        """Delete a file. Returns ``True`` if it existed."""
        normalized = self.normalize_path(path)
        if normalized in self._files:
            del self._files[normalized]
            return True
        return False

    def list_directory(self, path: str) -> list[str]:
        """List immediate children (files and subdirectories) under *path*.

        Returns basenames, not full paths. Directories are deduced from
        file paths and are suffixed with ``/``.
        """
        normalized = self.normalize_path(path)
        if not normalized.endswith("/"):
            normalized += "/"

        children: set[str] = set()
        for file_path in self._files:
            if not file_path.startswith(normalized):
                continue
            remainder = file_path[len(normalized) :]
            if not remainder:
                continue
            parts = remainder.split("/")
            if len(parts) == 1:
                children.add(parts[0])
            else:
                children.add(parts[0] + "/")

        return sorted(children)

    def glob(self, pattern: str) -> list[str]:
        """Match file paths against a glob pattern.

        Supports ``*``, ``**``, and ``?`` wildcards.
        Returns sorted list of matching absolute paths.
        """
        pattern = self.normalize_path(pattern)
        # Convert glob pattern to regex for proper ** handling
        regex = _glob_to_regex(pattern)
        compiled = re.compile(regex)
        matches = [p for p in sorted(self._files) if compiled.match(p)]
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_filter: str | None = None,
        ignore_case: bool = False,
    ) -> list[dict[str, str | int]]:
        """Search file contents for a regex pattern.

        Args:
            pattern: Regular expression to search for.
            path: Restrict search to files under this directory.
            glob_filter: Only search files matching this glob pattern.
            ignore_case: Case-insensitive matching.

        Returns:
            List of ``{"path": str, "line": int, "text": str}`` dicts.
        """
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            return []

        base = self.normalize_path(path) if path else "/"
        if not base.endswith("/"):
            base += "/"

        results: list[dict[str, str | int]] = []
        for file_path in sorted(self._files):
            if not file_path.startswith(base) and file_path != base.rstrip("/"):
                continue
            if glob_filter and not fnmatch.fnmatch(file_path, self.normalize_path(glob_filter)):
                continue

            content = self._files[file_path]
            for line_num, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    results.append({"path": file_path, "line": line_num, "text": line})

        return results

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Replace exact string occurrences in a file.

        Args:
            path: File to edit.
            old_string: Exact string to find.
            new_string: Replacement string.
            replace_all: If ``False``, *old_string* must appear exactly once.

        Returns:
            Number of replacements made.

        Raises:
            FileNotFoundError: File does not exist.
            ValueError: *old_string* not found, or found multiple times
                when ``replace_all=False``.
        """
        normalized = self.normalize_path(path)
        content = self._files.get(normalized)
        if content is None:
            raise FileNotFoundError(f"File not found: {path}")

        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"String not found in {path}")
        if not replace_all and count > 1:
            raise ValueError(
                f"Found {count} occurrences in {path}. "
                f"Use replace_all=True to replace all, or provide more context to make it unique."
            )

        if replace_all:
            self._files[normalized] = content.replace(old_string, new_string)
        else:
            self._files[normalized] = content.replace(old_string, new_string, 1)

        return count if replace_all else 1

    @property
    def files(self) -> dict[str, str]:
        """Read-only view of all files. Keys are absolute paths."""
        return dict(self._files)

    def __len__(self) -> int:
        return len(self._files)

    def __contains__(self, path: str) -> bool:
        return self.exists(path)
