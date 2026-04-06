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
import difflib
import json
import os
import time
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

_IMAGE_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_GLOB_DEFAULT_LIMIT = 100

# Read-state cache for file_unchanged dedup (cache_key → content_hash)
_read_state: dict[str, str] = {}

FILE_UNCHANGED_STUB = (
    "<system-reminder>The file has not changed since your last read.</system-reminder>"
)

# ---------------------------------------------------------------------------
# Quote normalization constants (ported from Claude Code utils.ts)
# ---------------------------------------------------------------------------

_LEFT_SINGLE_CURLY = "\u2018"   # '
_RIGHT_SINGLE_CURLY = "\u2019"  # '
_LEFT_DOUBLE_CURLY = "\u201c"   # "
_RIGHT_DOUBLE_CURLY = "\u201d"  # "

_CURLY_TO_STRAIGHT: dict[str, str] = {
    _LEFT_SINGLE_CURLY: "'",
    _RIGHT_SINGLE_CURLY: "'",
    _LEFT_DOUBLE_CURLY: '"',
    _RIGHT_DOUBLE_CURLY: '"',
}


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
    pages: str | None = Field(
        default=None,
        description=(
            'Page range for PDF files (e.g., "1-5", "3", "10-20"). '
            "Only applicable to PDF files."
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
        description="Number of lines to show before and after each match (rg -C).",
    )
    before_context: int | None = Field(
        default=None,
        alias="-B",
        description="Number of lines to show before each match (rg -B).",
    )
    after_context: int | None = Field(
        default=None,
        alias="-A",
        description="Number of lines to show after each match (rg -A).",
    )
    context_alias: int | None = Field(
        default=None,
        alias="-C",
        description="Alias for context.",
    )
    line_numbers: bool = Field(
        default=True,
        alias="-n",
        description="Show line numbers in output. Defaults to true.",
    )
    ignore_case: bool = Field(
        default=False,
        alias="-i",
        description="Case insensitive search (rg -i).",
    )
    type: str | None = Field(
        default=None,
        description=(
            "File type to search (rg --type). "
            "Common types: js, py, rust, go, java, etc."
        ),
    )
    head_limit: int = Field(
        default=250,
        description=(
            "Limit output to first N lines/entries. "
            "Defaults to 250. Pass 0 for unlimited."
        ),
    )
    offset: int = Field(
        default=0,
        description=(
            "Skip first N lines/entries before applying head_limit."
        ),
    )
    multiline: bool = Field(
        default=False,
        description=(
            "Enable multiline mode where . matches newlines "
            "and patterns can span lines."
        ),
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Helpers — line prefix stripping
# ---------------------------------------------------------------------------


def _format_file_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _strip_line_prefixes(raw: str) -> str:
    """Strip line-number prefixes from backend.read() output."""
    parts: list[str] = []
    for line in raw.splitlines(keepends=True):
        _, _, text = line.partition("\t")
        parts.append(text)
    return "".join(parts)


def _compute_structured_patch(  # noqa: C901
    old_content: str,
    new_content: str,
    context_lines: int = 3,
) -> list[dict[str, Any]]:
    """Compute structured patch hunks matching the reference schema.

    Returns list of dicts with: oldStart, oldLines, newStart, newLines, lines.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    hunks: list[dict[str, Any]] = []

    for group in difflib.SequenceMatcher(None, old_lines, new_lines).get_grouped_opcodes(
        context_lines,
    ):
        old_start = group[0][1] + 1  # 1-indexed
        old_end = group[-1][2]
        new_start = group[0][3] + 1
        new_end = group[-1][4]
        lines: list[str] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line in old_lines[i1:i2]:
                    lines.append(" " + line.rstrip("\n"))
            elif tag == "replace":
                for line in old_lines[i1:i2]:
                    lines.append("-" + line.rstrip("\n"))
                for line in new_lines[j1:j2]:
                    lines.append("+" + line.rstrip("\n"))
            elif tag == "delete":
                for line in old_lines[i1:i2]:
                    lines.append("-" + line.rstrip("\n"))
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    lines.append("+" + line.rstrip("\n"))
        hunks.append({
            "oldStart": old_start,
            "oldLines": old_end - old_start + 1,
            "newStart": new_start,
            "newLines": new_end - new_start + 1,
            "lines": lines,
        })
    return hunks


def _read_full_text(backend: Any, file_path: str) -> str:
    """Read full file content as text, stripping line-number prefixes."""
    raw = backend.read(file_path, limit=100_000)
    return _strip_line_prefixes(raw)


# ---------------------------------------------------------------------------
# Helpers — quote normalization
# ---------------------------------------------------------------------------


def _normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes for matching."""
    result = text
    for curly, straight in _CURLY_TO_STRAIGHT.items():
        result = result.replace(curly, straight)
    return result


def _is_opening_context(chars: list[str], index: int) -> bool:
    """Check if the character at index is in an opening quote context."""
    if index == 0:
        return True
    prev = chars[index - 1]
    return prev in (" ", "\t", "\n", "\r", "(", "[", "{", "\u2014", "\u2013")


def _apply_curly_double_quotes(text: str) -> str:
    """Replace straight double quotes with contextual curly double quotes."""
    chars = list(text)
    result: list[str] = []
    for i, ch in enumerate(chars):
        if ch == '"':
            result.append(
                _LEFT_DOUBLE_CURLY if _is_opening_context(chars, i)
                else _RIGHT_DOUBLE_CURLY,
            )
        else:
            result.append(ch)
    return "".join(result)


def _apply_curly_single_quotes(text: str) -> str:
    """Replace straight single quotes with contextual curly single quotes.

    Apostrophes in contractions (letter-'-letter) use right single curly.
    """
    import unicodedata

    chars = list(text)
    result: list[str] = []
    for i, ch in enumerate(chars):
        if ch == "'":
            prev = chars[i - 1] if i > 0 else ""
            nxt = chars[i + 1] if i < len(chars) - 1 else ""
            prev_is_letter = bool(prev and unicodedata.category(prev).startswith("L"))
            next_is_letter = bool(nxt and unicodedata.category(nxt).startswith("L"))
            if prev_is_letter and next_is_letter:
                result.append(_RIGHT_SINGLE_CURLY)
            elif _is_opening_context(chars, i):
                result.append(_LEFT_SINGLE_CURLY)
            else:
                result.append(_RIGHT_SINGLE_CURLY)
        else:
            result.append(ch)
    return "".join(result)


def _preserve_quote_style(
    old_string: str,
    actual_old_string: str,
    new_string: str,
) -> str:
    """Apply the file's curly quote style to new_string when normalization was used."""
    if old_string == actual_old_string:
        return new_string

    has_double = (
        _LEFT_DOUBLE_CURLY in actual_old_string
        or _RIGHT_DOUBLE_CURLY in actual_old_string
    )
    has_single = (
        _LEFT_SINGLE_CURLY in actual_old_string
        or _RIGHT_SINGLE_CURLY in actual_old_string
    )

    if not has_double and not has_single:
        return new_string

    result = new_string
    if has_double:
        result = _apply_curly_double_quotes(result)
    if has_single:
        result = _apply_curly_single_quotes(result)
    return result


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """Find the actual string in file, falling back to quote-normalized matching.

    Returns the string as it appears in the file, or None if not found.
    """
    if search_string in file_content:
        return search_string
    # Try quote-normalized matching
    normalized_content = _normalize_quotes(file_content)
    normalized_search = _normalize_quotes(search_string)
    if normalized_search in normalized_content:
        # Find the position in normalized content and extract from original
        pos = normalized_content.index(normalized_search)
        return file_content[pos : pos + len(normalized_search)]
    return None


# ---------------------------------------------------------------------------
# Helpers — trailing whitespace stripping
# ---------------------------------------------------------------------------


def _strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace per line, preserving line endings."""
    import re

    lines = re.split(r"(\r\n|\n|\r)", text)
    result: list[str] = []
    for i, part in enumerate(lines):
        if i % 2 == 0:
            result.append(part.rstrip())
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# Helpers — grep
# ---------------------------------------------------------------------------


def _format_limit_info(shown: int, total: int, head_limit: int) -> str:
    """Format pagination info matching the reference's formatLimitInfo()."""
    if shown >= total:
        return ""
    return f"\n(showing {shown} of {total} results, use offset to paginate)"


def _grep_files_with_matches(
    results: list[dict[str, Any]],
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    paths = list(dict.fromkeys(r["path"] for r in results))
    entries = paths[:head_limit] if head_limit else paths
    truncated = len(paths) > len(entries)
    content = f"Found {len(paths)} file(s)\n" + "\n".join(entries)
    if truncated:
        content += _format_limit_info(len(entries), len(paths), head_limit)
    artifact: dict[str, Any] = {
        "mode": "files_with_matches",
        "numFiles": len(paths),
        "filenames": entries,
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


def _grep_count(
    results: list[dict[str, Any]],
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["path"]] = counts.get(r["path"], 0) + 1
    items = list(counts.items())
    entries = items[:head_limit] if head_limit else items
    truncated = len(items) > len(entries)
    content = "\n".join(f"{p}: {c} match(es)" for p, c in entries)
    if truncated:
        content += _format_limit_info(len(entries), len(items), head_limit)
    total_matches = sum(c for _, c in entries)
    artifact: dict[str, Any] = {
        "mode": "count",
        "numFiles": len(items),
        "numMatches": total_matches,
        "filenames": [],
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


def _grep_multiline(
    backend: Any,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool = False,
) -> list[dict[str, Any]]:
    """Full-file regex search with re.DOTALL for cross-line patterns."""
    import re

    flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
    regex = re.compile(pattern, flags)
    file_paths = backend.glob(glob or "**/*", path=path or "/")
    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        try:
            content = _read_full_text(backend, file_path)
            for match in regex.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                matched_text = match.group()
                for i, line_text in enumerate(matched_text.split("\n")):
                    results.append({
                        "path": file_path,
                        "line": line_num + i,
                        "text": line_text,
                    })
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
    return results


def _format_grep_with_context(
    backend: Any,
    results: list[dict[str, Any]],
    before: int = 0,
    after: int = 0,
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
                stripped = []
                for line in raw.splitlines():
                    _, _, content = line.partition("\t")
                    stripped.append(content)
                file_lines_cache[file_path] = stripped
            except (FileNotFoundError, OSError):
                file_lines_cache[file_path] = []

        lines = file_lines_cache[file_path]
        start = max(0, line_num - 1 - before)
        end = min(len(lines), line_num + after)
        for i in range(start, end):
            prefix = ":" if i == line_num - 1 else "-"
            output.append(f"{file_path}{prefix}{i + 1}: {lines[i]}")
        output.append("--")

    return output


_TYPE_GLOB_MAP: dict[str, str] = {
    "py": "*.py", "js": "*.js", "ts": "*.ts", "tsx": "*.tsx",
    "jsx": "*.jsx", "java": "*.java", "go": "*.go", "rs": "*.rs",
    "rust": "*.rs", "rb": "*.rb", "c": "*.c", "cpp": "*.cpp",
    "h": "*.h", "cs": "*.cs", "php": "*.php", "swift": "*.swift",
    "kt": "*.kt", "scala": "*.scala", "sh": "*.sh", "yaml": "*.yaml",
    "yml": "*.yml", "json": "*.json", "xml": "*.xml", "html": "*.html",
    "css": "*.css", "md": "*.md", "sql": "*.sql", "r": "*.r",
}


def _resolve_grep_glob(glob: str | None, file_type: str | None) -> str | None:
    """Resolve effective glob from explicit glob or type filter."""
    if glob is not None:
        return glob
    if file_type is not None:
        return _TYPE_GLOB_MAP.get(file_type, f"*.{file_type}")
    return None


def _apply_offset_to_results(
    results: list[dict[str, Any]], offset: int, output_mode: str,
) -> list[dict[str, Any]]:
    """Skip first N entries, scoped by output mode."""
    if offset <= 0:
        return results
    if output_mode in ("files_with_matches", "count"):
        paths = list(dict.fromkeys(r["path"] for r in results))[offset:]
        path_set = set(paths)
        return [r for r in results if r["path"] in path_set]
    return results[offset:]


def _resolve_grep_context(
    context: int | None,
    context_alias: int | None,
    before_context: int | None,
    after_context: int | None,
) -> tuple[int | None, int | None]:
    """Resolve context precedence: context > -C > -B/-A individually.

    Returns (effective_before, effective_after).
    """
    # context takes full precedence
    if context is not None:
        return context, context
    # -C alias next
    if context_alias is not None:
        return context_alias, context_alias
    # Individual -B/-A only when neither context nor -C is set
    return before_context, after_context


def _format_content_results(
    backend: Any,
    results: list[dict[str, Any]],
    before: int | None,
    after: int | None,
    line_numbers: bool,
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format grep results in content mode. Returns (content, artifact)."""
    has_context = (before and before > 0) or (after and after > 0)
    if has_context:
        lines = _format_grep_with_context(
            backend, results, before=before or 0, after=after or 0,
        )
    elif line_numbers:
        lines = [f"{r['path']}:{r['line']}: {r['text']}" for r in results]
    else:
        lines = [f"{r['path']}: {r['text']}" for r in results]
    total_lines = len(lines)
    truncated = head_limit > 0 and total_lines > head_limit
    if head_limit:
        lines = lines[:head_limit]
    content = "\n".join(lines)
    if truncated:
        content += _format_limit_info(len(lines), total_lines, head_limit)
    artifact: dict[str, Any] = {
        "mode": "content",
        "numFiles": len(set(r["path"] for r in results)),
        "numLines": len(lines),
        "filenames": [],
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


# ---------------------------------------------------------------------------
# Helpers — edit
# ---------------------------------------------------------------------------


def _handle_empty_old_string(
    backend: Any, file_path: str, new_string: str,
) -> tuple[str, dict[str, Any]]:
    """Handle edit with empty old_string — file creation or filling empty file."""
    try:
        content = backend.read(file_path, limit=1)
        if _strip_line_prefixes(content).strip():
            raise ToolException(
                f"File {file_path} is not empty. "
                f"Cannot use empty old_string on a non-empty file."
            )
    except FileNotFoundError:
        pass  # File doesn't exist — will create
    backend.write(file_path, new_string)
    message = f"The file {file_path} has been updated successfully."
    artifact: dict[str, Any] = {
        "filePath": file_path,
        "oldString": "",
        "newString": new_string,
        "replaceAll": False,
        "originalFile": None,
    }
    return message, artifact


def _resolve_trailing_newline(backend: Any, file_path: str, old_string: str) -> str:
    """When deleting text, auto-include trailing newline to prevent orphan blanks."""
    if old_string.endswith("\n"):
        return old_string
    try:
        file_text = _read_full_text(backend, file_path)
        if old_string + "\n" in file_text:
            return old_string + "\n"
    except (FileNotFoundError, OSError):
        pass
    return old_string


# ---------------------------------------------------------------------------
# Tool builders — Read
# ---------------------------------------------------------------------------


def _read_notebook(backend: Any, file_path: str) -> tuple[str, dict[str, Any]]:
    """Read a Jupyter notebook and return formatted cell contents."""
    data = backend.read_bytes(file_path)
    notebook = json.loads(data.decode("utf-8"))
    cells = notebook.get("cells", [])
    parts: list[str] = []
    structured_cells: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source_lines = cell.get("source", [])
        source = "".join(source_lines) if isinstance(source_lines, list) else source_lines
        parts.append(f"\n--- Cell {i + 1} ({cell_type}) ---")
        parts.append(source)
        cell_outputs: list[str] = []
        for output in cell.get("outputs", []):
            _format_notebook_output(output, parts, cell_outputs)
        structured_cells.append({
            "cell_type": cell_type,
            "source": source,
            "outputs": cell_outputs,
        })
    content = "\n".join(parts)
    return content, {
        "type": "notebook",
        "filePath": file_path,
        "cells": structured_cells,
    }


def _format_notebook_output(
    output: dict[str, Any],
    parts: list[str],
    cell_outputs: list[str],
) -> None:
    """Format a single notebook cell output and append to parts."""
    output_type = output.get("output_type", "")
    if output_type == "stream":
        text = "".join(output.get("text", []))
        parts.append(f"[Output]\n{text}")
        cell_outputs.append(text)
    elif output_type in ("execute_result", "display_data"):
        text_data = output.get("data", {}).get("text/plain", [])
        text = "".join(text_data) if isinstance(text_data, list) else text_data
        if text:
            parts.append(f"[Output]\n{text}")
            cell_outputs.append(text)
    elif output_type == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        parts.append(f"[Error: {ename}] {evalue}")
        cell_outputs.append(f"{ename}: {evalue}")


def _detect_image_dimensions(data: bytes) -> dict[str, int] | None:
    """Detect image dimensions. Returns dict with width/height or None."""
    try:
        import io

        from PIL import Image as PILImage  # type: ignore[import-not-found]

        img = PILImage.open(io.BytesIO(data))
        w, h = img.size
        img.close()
        return {"originalWidth": w, "originalHeight": h}
    except Exception:  # noqa: BLE001
        return None


def _read_image(backend: Any, file_path: str, ext: str) -> tuple[str, dict[str, Any]]:
    """Read an image file and return as multimodal content block."""
    data = backend.read_bytes(file_path)
    mime_type = _IMAGE_MIME_MAP.get(ext, "application/octet-stream")
    encoded = base64.b64encode(data).decode("ascii")

    # LLM content: multimodal image block (matches CC FileReadTool.ts:654-668)
    content = json.dumps([{
        "type": "image",
        "source": {
            "type": "base64",
            "data": encoded,
            "media_type": mime_type,
        },
    }])

    dimensions = _detect_image_dimensions(data)
    artifact: dict[str, Any] = {
        "type": "image",
        "filePath": file_path,
        "base64": encoded,
        "mediaType": mime_type,
        "originalSize": len(data),
    }
    if dimensions:
        artifact["dimensions"] = dimensions
    return content, artifact


def _parse_page_range(pages: str) -> tuple[int, int]:
    """Parse a page range string like '1-5', '3', '10-20'. Returns (start, end) 0-indexed."""
    pages = pages.strip()
    if "-" in pages:
        parts = pages.split("-", 1)
        start = int(parts[0]) - 1
        end = int(parts[1])
    else:
        start = int(pages) - 1
        end = int(pages)
    return max(0, start), end


def _read_pdf(
    backend: Any, file_path: str, pages: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Read a PDF file and return base64 content, optionally extracting pages."""
    data = backend.read_bytes(file_path)

    if pages is not None:
        return _read_pdf_pages(data, file_path, pages)

    encoded = base64.b64encode(data).decode("ascii")
    size_str = _format_file_size(len(data))
    content = f"PDF file read: {file_path} ({size_str})"
    artifact: dict[str, Any] = {
        "type": "pdf",
        "filePath": file_path,
        "base64": encoded,
        "originalSize": len(data),
    }
    return content, artifact


def _read_pdf_pages(
    data: bytes, file_path: str, pages: str,
) -> tuple[str, dict[str, Any]]:
    """Extract specific pages from a PDF. Requires pymupdf."""
    try:
        import fitz  # type: ignore[import-not-found]  # pymupdf (optional)
    except ImportError:
        # Graceful degradation: return full PDF with note
        encoded = base64.b64encode(data).decode("ascii")
        size_str = _format_file_size(len(data))
        content = (
            f"PDF file read: {file_path} ({size_str}). "
            f"Page extraction requires 'pip install pymupdf'."
        )
        return content, {
            "type": "pdf",
            "filePath": file_path,
            "base64": encoded,
            "originalSize": len(data),
        }

    start, end = _parse_page_range(pages)
    doc = fitz.open(stream=data, filetype="pdf")
    page_count = min(end, len(doc)) - start
    page_texts: list[str] = []
    for i in range(start, min(end, len(doc))):
        page_texts.append(f"--- Page {i + 1} ---\n{doc[i].get_text()}")
    doc.close()

    size_str = _format_file_size(len(data))
    content = (
        f"PDF pages extracted: {page_count} page(s) from {file_path} ({size_str})\n"
        + "\n".join(page_texts)
    )
    return content, {
        "type": "parts",
        "filePath": file_path,
        "originalSize": len(data),
        "count": page_count,
    }


def _get_content_hash(backend: Any, file_path: str) -> str:
    """Get a content hash for dedup. Uses file content since mtime isn't in protocol."""
    import hashlib

    try:
        data = backend.read_bytes(file_path)
        return hashlib.md5(data).hexdigest()  # noqa: S324
    except (FileNotFoundError, OSError):
        return ""


def _check_file_unchanged(
    backend: Any, file_path: str, offset: int, limit: int,
) -> bool:
    """Check if a file is unchanged since last read with same offset/limit."""
    cache_key = f"{file_path}:{offset}:{limit}"
    if cache_key not in _read_state:
        return False
    cached_hash = _read_state[cache_key]
    current_hash = _get_content_hash(backend, file_path)
    return bool(current_hash and current_hash == cached_hash)


def _update_read_state(
    backend: Any, file_path: str, offset: int, limit: int,
) -> None:
    """Cache the file's content hash for dedup."""
    cache_key = f"{file_path}:{offset}:{limit}"
    content_hash = _get_content_hash(backend, file_path)
    if content_hash:
        _read_state[cache_key] = content_hash


def _read_text(
    backend: Any, file_path: str, offset: int, limit: int,
) -> tuple[str, dict[str, Any]]:
    """Read a text file with line numbers."""
    # file_unchanged dedup (matches CC FileReadTool.ts:326-329)
    if _check_file_unchanged(backend, file_path, offset, limit):
        return FILE_UNCHANGED_STUB, {
            "type": "file_unchanged",
            "filePath": file_path,
        }

    raw = str(backend.read(file_path, offset=offset, limit=limit))
    text = _strip_line_prefixes(raw)
    num_lines = len(text.splitlines()) if text else 0

    # Count total lines in file for the artifact
    total_raw = str(backend.read(file_path, limit=100_000))
    total_text = _strip_line_prefixes(total_raw)
    total_lines = len(total_text.splitlines()) if total_text else 0

    # Empty/short file warnings (matches CC FileReadTool.ts:703-707)
    if not text:
        if total_lines == 0:
            content = (
                "<system-reminder>Warning: the file exists but "
                "the contents are empty.</system-reminder>"
            )
        else:
            content = (
                f"<system-reminder>Warning: the file exists but is shorter "
                f"than the provided offset ({offset + 1}). "
                f"The file has {total_lines} lines.</system-reminder>"
            )
    else:
        content = raw

    _update_read_state(backend, file_path, offset, limit)

    artifact: dict[str, Any] = {
        "type": "text",
        "filePath": file_path,
        "content": text,
        "numLines": num_lines,
        "startLine": offset + 1,
        "totalLines": total_lines,
    }
    return content, artifact


def _build_read(backend: Any) -> BaseTool:
    def read(
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        pages: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Read a file with type-aware dispatch."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in NOTEBOOK_EXTENSIONS:
            return _read_notebook(backend, file_path)

        if ext in BINARY_EXTENSIONS:
            if ext in IMAGE_EXTENSIONS:
                return _read_image(backend, file_path, ext)
            if ext in PDF_EXTENSIONS:
                return _read_pdf(backend, file_path, pages=pages)
            raise ToolException(
                f"Cannot read binary {ext} file. "
                f"Use appropriate tools for binary file analysis."
            )

        return _read_text(backend, file_path, offset, limit)

    return StructuredTool.from_function(
        func=read,
        name="Read",
        description=(
            "Read a file. Results returned with line numbers. "
            "Use offset and limit for large files. "
            "Supports text files, images (png/jpg/gif/webp), PDFs, and Jupyter notebooks."
        ),
        args_schema=_ReadInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Tool builders — Write
# ---------------------------------------------------------------------------


def _build_write(backend: Any) -> BaseTool:
    def write(file_path: str, content: str) -> tuple[str, dict[str, Any]]:
        """Write content to a file."""
        is_new = False
        original_file: str | None = None
        try:
            original_file = _read_full_text(backend, file_path)
        except (FileNotFoundError, OSError):
            is_new = True

        backend.write(file_path, content)
        op_type = "create" if is_new else "update"

        if is_new:
            message = f"File created successfully at: {file_path}"
        else:
            message = f"The file {file_path} has been updated successfully."

        patch = (
            _compute_structured_patch(original_file, content)
            if original_file is not None
            else []
        )
        artifact: dict[str, Any] = {
            "type": op_type,
            "filePath": file_path,
            "content": content,
            "structuredPatch": patch,
            "originalFile": original_file,
        }
        return message, artifact

    return StructuredTool.from_function(
        func=write,
        name="Write",
        description=(
            "Write content to a file, creating or overwriting it. Prefer Edit for modifications."
        ),
        args_schema=_WriteInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Tool builders — Edit
# ---------------------------------------------------------------------------


def _build_edit(backend: Any) -> BaseTool:  # noqa: C901
    def edit(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Perform exact string replacement in a file."""
        if old_string == "":
            return _handle_empty_old_string(backend, file_path, new_string)

        # Read original file for quote normalization and artifact
        try:
            original_file = _read_full_text(backend, file_path)
        except FileNotFoundError as exc:
            raise ToolException(str(exc)) from exc

        # Strip trailing whitespace from new_string (except markdown)
        effective_new = new_string
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".md", ".mdx"):
            effective_new = _strip_trailing_whitespace(new_string)

        # Quote normalization: find actual string in file
        actual_old = _find_actual_string(original_file, old_string)
        if actual_old is None:
            raise ToolException(f"String not found in {file_path}.")

        # Preserve the file's quote style in new_string
        effective_new = _preserve_quote_style(old_string, actual_old, effective_new)

        # Trailing newline stripping on delete
        effective_old = actual_old
        if (
            effective_new == ""
            and not actual_old.endswith("\n")
            and actual_old + "\n" in original_file
        ):
            effective_old = actual_old + "\n"

        try:
            result = backend.edit(
                file_path, effective_old, effective_new, replace_all=replace_all,
            )
            count = result.get("replacements", 0) if isinstance(result, dict) else result
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        if count == 0:
            raise ToolException(f"String not found in {file_path}.")

        # Read updated content for patch computation
        updated_file = _read_full_text(backend, file_path)
        patch = _compute_structured_patch(original_file, updated_file)

        artifact: dict[str, Any] = {
            "filePath": file_path,
            "oldString": old_string,
            "newString": new_string,
            "replaceAll": replace_all,
            "originalFile": original_file,
            "structuredPatch": patch,
        }

        if replace_all:
            message = (
                f"The file {file_path} has been updated. "
                f"All occurrences were successfully replaced."
            )
        else:
            message = f"The file {file_path} has been updated successfully."
        return message, artifact

    return StructuredTool.from_function(
        func=edit,
        name="Edit",
        description=(
            "Perform exact string replacements in a file. "
            "Fails if old_string is not unique (use replace_all for multiple)."
        ),
        args_schema=_EditInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Tool builders — Glob
# ---------------------------------------------------------------------------


def _build_glob(backend: Any) -> BaseTool:
    def glob(pattern: str, path: str | None = None) -> tuple[str, dict[str, Any]]:
        """Find files matching a glob pattern."""
        start = time.monotonic()
        all_matches = backend.glob(pattern, path=path or "/")
        duration_ms = round((time.monotonic() - start) * 1000)

        if not all_matches:
            return "No files found", {
                "numFiles": 0,
                "filenames": [],
                "truncated": False,
                "durationMs": duration_ms,
            }

        truncated = len(all_matches) > _GLOB_DEFAULT_LIMIT
        matches = all_matches[:_GLOB_DEFAULT_LIMIT]
        content = "\n".join(matches)
        if truncated:
            content += (
                "\n(Results are truncated. "
                "Consider using a more specific path or pattern.)"
            )
        artifact: dict[str, Any] = {
            "numFiles": len(all_matches),
            "filenames": matches,
            "truncated": truncated,
            "durationMs": duration_ms,
        }
        return content, artifact

    return StructuredTool.from_function(
        func=glob,
        name="Glob",
        description=(
            "Fast file pattern matching. "
            'Supports patterns like "**/*.py". Returns matching file paths.'
        ),
        args_schema=_GlobInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Tool builders — Grep
# ---------------------------------------------------------------------------


def _build_grep(backend: Any) -> BaseTool:
    def grep(  # noqa: PLR0913
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        context: int | None = None,
        before_context: int | None = None,
        after_context: int | None = None,
        context_alias: int | None = None,
        line_numbers: bool = True,
        ignore_case: bool = False,
        type: str | None = None,
        head_limit: int = 250,
        offset: int = 0,
        multiline: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Search file contents for a regex pattern."""
        effective_before, effective_after = _resolve_grep_context(
            context, context_alias, before_context, after_context,
        )
        effective_glob = _resolve_grep_glob(glob, type)

        if multiline:
            results = _grep_multiline(
                backend, pattern, path=path, glob=effective_glob,
                ignore_case=ignore_case,
            )
        else:
            results = backend.grep(
                pattern, path=path, glob=effective_glob, ignore_case=ignore_case,
            )
        if not results:
            return "No matches found.", {
                "mode": output_mode,
                "numFiles": 0,
                "filenames": [],
            }

        results = _apply_offset_to_results(results, offset, output_mode)

        if output_mode == "files_with_matches":
            return _grep_files_with_matches(results, head_limit, offset)
        if output_mode == "count":
            return _grep_count(results, head_limit, offset)
        return _format_content_results(
            backend, results, effective_before, effective_after,
            line_numbers, head_limit, offset,
        )

    return StructuredTool.from_function(
        func=grep,
        name="Grep",
        description=(
            "Search file contents for a regex pattern. "
            'Output modes: "content", "files_with_matches" (default), "count".'
        ),
        args_schema=_GrepInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
