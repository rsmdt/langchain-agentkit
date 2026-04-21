"""File-operation index — what files did the conversation touch?

During compaction we scan the to-be-discarded prefix for ``Read`` /
``Write`` / ``Edit`` tool calls and extract the paths. The resulting
``<read-files>`` / ``<modified-files>`` block is appended to the summary
so the post-compaction model can quickly reconstitute "what did we
touch" without re-reading everything.

Files appearing in both reads and writes are classified as modified
only — the modified list supersedes the read list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Iterable

# Tool names that contribute file paths to the index. Matches the
# built-in FilesystemExtension tool set.
_READ_TOOLS = frozenset({"Read"})
_WRITE_TOOLS = frozenset({"Write"})
_EDIT_TOOLS = frozenset({"Edit"})


@dataclass(slots=True)
class FileOps:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)

    def extend(self, other: FileOps) -> None:
        self.read.update(other.read)
        self.written.update(other.written)
        self.edited.update(other.edited)


def extract_file_ops(messages: Iterable[Any]) -> FileOps:
    """Scan ``messages`` for Read/Write/Edit tool calls and collect paths."""
    ops = FileOps()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name", "")
            args = call.get("args") or {}
            if not isinstance(args, dict):
                continue
            path = args.get("path") or args.get("file_path")
            if not isinstance(path, str) or not path:
                continue
            if name in _READ_TOOLS:
                ops.read.add(path)
            elif name in _WRITE_TOOLS:
                ops.written.add(path)
            elif name in _EDIT_TOOLS:
                ops.edited.add(path)
    return ops


def compute_file_lists(ops: FileOps) -> tuple[list[str], list[str]]:
    """Return ``(read_only, modified)`` lists derived from ``ops``."""
    modified = ops.written | ops.edited
    read_only = sorted(p for p in ops.read if p not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Render XML-style sections for inclusion in the summary."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)
