"""Filesystem tools operating on a BackendProtocol.

Provides five tools: ``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``.
Parameters and descriptions match the Claude Code tool API.

Usage::

    from langchain_agentkit.backend import OSBackend
    from langchain_agentkit.tools.filesystem import create_filesystem_tools

    backend = OSBackend("./workspace")
    tools = create_filesystem_tools(backend)
"""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# File type constants (ported from Claude Code constants/files.ts)
# ---------------------------------------------------------------------------

BINARY_EXTENSIONS: frozenset[str] = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    # Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    # Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    # Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    # Flash
    ".swf", ".fla",
    # Lock/profiling data
    ".lockb", ".dat", ".data",
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
})

PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

NOTEBOOK_EXTENSIONS: frozenset[str] = frozenset({".ipynb"})


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _ReadInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to read.",
    )
    offset: int = Field(
        default=0,
        description=(
            "The line number to start reading from. "
            "Only provide if the file is too large to read at once."
        ),
    )
    limit: int = Field(
        default=2000,
        description=(
            "The number of lines to read. Only provide if the file is too large to read at once."
        ),
    )


class _WriteInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to write.",
    )
    content: str = Field(
        description="The content to write to the file.",
    )


class _EditInput(BaseModel):
    file_path: str = Field(
        description="The absolute path to the file to modify.",
    )
    old_string: str = Field(
        description="The text to replace.",
    )
    new_string: str = Field(
        description=("The text to replace it with (must be different from old_string)."),
    )
    replace_all: bool = Field(
        default=False,
        description=(
            "Replace all occurrences of old_string (default false). "
            "If false, old_string must be unique in the file."
        ),
    )


class _GlobInput(BaseModel):
    pattern: str = Field(
        description='The glob pattern to match files against (e.g., "**/*.py").',
    )
    path: str | None = Field(
        default=None,
        description=(
            "The directory to search in. If not specified, searches the entire filesystem."
        ),
    )


class _GrepInput(BaseModel):
    pattern: str = Field(
        description="The regular expression pattern to search for in file contents.",
    )
    path: str | None = Field(
        default=None,
        description="File or directory to search in. Defaults to entire filesystem.",
    )
    glob: str | None = Field(
        default=None,
        description='Glob pattern to filter files (e.g., "*.py", "**/*.tsx").',
    )
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="files_with_matches",
        description=(
            "Output mode: "
            '"content" shows matching lines, '
            '"files_with_matches" shows only file paths (default), '
            '"count" shows match counts per file.'
        ),
    )
    context: int | None = Field(
        default=None,
        description="Number of lines to show before and after each match.",
    )
    ignore_case: bool = Field(
        default=False,
        description="Case insensitive search.",
    )
    head_limit: int = Field(
        default=0,
        description=("Limit output to first N lines/entries. 0 means unlimited (default)."),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grep_files_with_matches(
    results: list[dict[str, Any]],
    head_limit: int,
) -> str:
    paths = list(dict.fromkeys(r["path"] for r in results))
    entries = paths[:head_limit] if head_limit else paths
    output = "\n".join(entries)
    remaining = len(paths) - len(entries)
    if remaining > 0:
        output += f"\n... ({remaining} more files)"
    return output


def _grep_count(results: list[dict[str, Any]], head_limit: int) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["path"]] = counts.get(r["path"], 0) + 1
    items = list(counts.items())
    entries = items[:head_limit] if head_limit else items
    output = "\n".join(f"{p}: {c} match(es)" for p, c in entries)
    remaining = len(items) - len(entries)
    if remaining > 0:
        output += f"\n... ({remaining} more files)"
    return output


def _format_grep_with_context(
    backend: Any,
    results: list[dict[str, Any]],
    context: int,
) -> list[str]:
    """Format grep results with surrounding context lines from the backend."""
    file_lines_cache: dict[str, list[str]] = {}
    output: list[str] = []

    for r in results:
        file_path = r["path"]
        line_num = r["line"]

        if file_path not in file_lines_cache:
            try:
                raw = backend.read(file_path, limit=100_000)
                # Strip line-number prefixes from backend.read() output
                stripped = []
                for line in raw.splitlines():
                    _, _, content = line.partition("\t")
                    stripped.append(content)
                file_lines_cache[file_path] = stripped
            except (FileNotFoundError, OSError):
                file_lines_cache[file_path] = []

        lines = file_lines_cache[file_path]
        start = max(0, line_num - 1 - context)
        end = min(len(lines), line_num + context)
        for i in range(start, end):
            prefix = ":" if i == line_num - 1 else "-"
            output.append(f"{file_path}{prefix}{i + 1}: {lines[i]}")
        output.append("--")

    return output


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def _read_image(backend: Any, file_path: str) -> str:
    """Read an image file and return base64-encoded content with metadata."""
    data = backend.read_bytes(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")
    encoded = base64.b64encode(data).decode("ascii")
    return (
        f"[Image: {file_path}]\n"
        f"Type: {mime_type}\n"
        f"Size: {len(data)} bytes\n"
        f"Base64: {encoded}"
    )


def _read_pdf(backend: Any, file_path: str) -> str:
    """Read a PDF file and return base64-encoded content."""
    data = backend.read_bytes(file_path)
    encoded = base64.b64encode(data).decode("ascii")
    return (
        f"[PDF: {file_path}]\n"
        f"Size: {len(data)} bytes\n"
        f"Base64: {encoded}"
    )


def _read_notebook(backend: Any, file_path: str) -> str:
    """Read a Jupyter notebook and return formatted cell contents."""
    data = backend.read_bytes(file_path)
    notebook = json.loads(data.decode("utf-8"))
    cells = notebook.get("cells", [])
    parts: list[str] = [f"[Notebook: {file_path}]"]
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source_lines = cell.get("source", [])
        source = "".join(source_lines) if isinstance(source_lines, list) else source_lines
        parts.append(f"\n--- Cell {i + 1} ({cell_type}) ---")
        parts.append(source)
        outputs = cell.get("outputs", [])
        for output in outputs:
            output_type = output.get("output_type", "")
            if output_type == "stream":
                text = "".join(output.get("text", []))
                parts.append(f"[Output]\n{text}")
            elif output_type in ("execute_result", "display_data"):
                text_data = output.get("data", {}).get("text/plain", [])
                text = "".join(text_data) if isinstance(text_data, list) else text_data
                if text:
                    parts.append(f"[Output]\n{text}")
            elif output_type == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                parts.append(f"[Error: {ename}] {evalue}")
    return "\n".join(parts)


def _build_read(backend: Any) -> BaseTool:
    def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with type-aware dispatch."""
        ext = os.path.splitext(file_path)[1].lower()

        # Notebook (check before binary blocklist — .ipynb is not in it but check explicitly)
        if ext in NOTEBOOK_EXTENSIONS:
            return _read_notebook(backend, file_path)

        # Binary check with carve-outs for natively supported formats
        if ext in BINARY_EXTENSIONS:
            if ext in IMAGE_EXTENSIONS:
                return _read_image(backend, file_path)
            if ext in PDF_EXTENSIONS:
                return _read_pdf(backend, file_path)
            raise ToolException(
                f"Cannot read binary {ext} file. "
                f"Use appropriate tools for binary file analysis."
            )

        # Default: text with line numbers
        return str(backend.read(file_path, offset=offset, limit=limit))

    return StructuredTool.from_function(
        func=read,
        name="Read",
        description=(
            "Read a file. Results returned with line numbers. "
            "Use offset and limit for large files. "
            "Supports text files, images (png/jpg/gif/webp), PDFs, and Jupyter notebooks."
        ),
        args_schema=_ReadInput,
        handle_tool_error=True,
    )


def _build_write(backend: Any) -> BaseTool:
    def write(file_path: str, content: str) -> str:
        """Write content to a file."""
        result = backend.write(file_path, content)
        return f"Wrote {result.get('bytes_written', len(content))} bytes to {file_path}"

    return StructuredTool.from_function(
        func=write,
        name="Write",
        description=(
            "Write content to a file, creating or overwriting it. Prefer Edit for modifications."
        ),
        args_schema=_WriteInput,
        handle_tool_error=True,
    )


def _build_edit(backend: Any) -> BaseTool:
    def edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Perform exact string replacement in a file."""
        try:
            result = backend.edit(file_path, old_string, new_string, replace_all=replace_all)
            count = result.get("replacements", 0) if isinstance(result, dict) else result
        except (FileNotFoundError, ValueError) as exc:
            raise ToolException(str(exc)) from exc
        return f"Replaced {count} occurrence(s) in {file_path}"

    return StructuredTool.from_function(
        func=edit,
        name="Edit",
        description=(
            "Perform exact string replacements in a file. "
            "Fails if old_string is not unique (use replace_all for multiple)."
        ),
        args_schema=_EditInput,
        handle_tool_error=True,
    )


def _build_glob(backend: Any) -> BaseTool:
    def glob(pattern: str, path: str | None = None) -> str:
        """Find files matching a glob pattern."""
        matches = backend.glob(pattern, path=path or "/")
        if not matches:
            return "No files matched."
        return "\n".join(matches)

    return StructuredTool.from_function(
        func=glob,
        name="Glob",
        description=(
            "Fast file pattern matching. "
            'Supports patterns like "**/*.py". Returns matching file paths.'
        ),
        args_schema=_GlobInput,
        handle_tool_error=True,
    )


def _build_grep(backend: Any) -> BaseTool:
    def grep(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        context: int | None = None,
        ignore_case: bool = False,
        head_limit: int = 0,
    ) -> str:
        """Search file contents for a regex pattern."""
        results = backend.grep(pattern, path=path, glob=glob, ignore_case=ignore_case)
        if not results:
            return "No matches found."
        if output_mode == "files_with_matches":
            return _grep_files_with_matches(results, head_limit)
        elif output_mode == "count":
            return _grep_count(results, head_limit)
        else:
            if context and context > 0:
                lines = _format_grep_with_context(backend, results, context)
            else:
                lines = [f"{r['path']}:{r['line']}: {r['text']}" for r in results]
            if head_limit:
                lines = lines[:head_limit]
            return "\n".join(lines)

    return StructuredTool.from_function(
        func=grep,
        name="Grep",
        description=(
            "Search file contents for a regex pattern. "
            'Output modes: "content", "files_with_matches" (default), "count".'
        ),
        args_schema=_GrepInput,
        handle_tool_error=True,
    )


def create_filesystem_tools(backend: Any) -> list[BaseTool]:
    """Create filesystem tools backed by a BackendProtocol.

    Returns five tools: ``[Read, Write, Edit, Glob, Grep]``,
    all operating on the given backend.

    Args:
        backend: A BackendProtocol implementation.
    """
    return [
        _build_read(backend),
        _build_write(backend),
        _build_edit(backend),
        _build_glob(backend),
        _build_grep(backend),
    ]
