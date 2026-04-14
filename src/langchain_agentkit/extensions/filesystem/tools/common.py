"""Shared constants, schemas, and helpers for filesystem tools."""

from __future__ import annotations

import difflib
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# File type constants
# ---------------------------------------------------------------------------

BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".tiff",
        ".tif",
        # Video
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".wmv",
        ".flv",
        ".m4v",
        ".mpeg",
        ".mpg",
        # Audio
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".aac",
        ".m4a",
        ".wma",
        ".aiff",
        ".opus",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".xz",
        ".z",
        ".tgz",
        ".iso",
        # Executables/binaries
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".o",
        ".a",
        ".obj",
        ".lib",
        ".app",
        ".msi",
        ".deb",
        ".rpm",
        # Documents
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # Bytecode / VM artifacts
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".node",
        ".wasm",
        ".rlib",
        # Database files
        ".sqlite",
        ".sqlite3",
        ".db",
        ".mdb",
        ".idx",
        # Design / 3D
        ".psd",
        ".ai",
        ".eps",
        ".sketch",
        ".fig",
        ".xd",
        ".blend",
        ".3ds",
        ".max",
        # Flash
        ".swf",
        ".fla",
        # Lock/profiling data
        ".lockb",
        ".dat",
        ".data",
    }
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    }
)

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

FILE_UNCHANGED_STUB = (
    "<system-reminder>The file has not changed since your last read.</system-reminder>"
)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _ReadInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read.")
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
            'Page range for PDF files (e.g., "1-5", "3", "10-20"). Only applicable to PDF files.'
        ),
    )


class _WriteInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to write.")
    content: str = Field(description="The content to write to the file.")


class _EditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to modify.")
    old_string: str = Field(description="The text to replace.")
    new_string: str = Field(
        description="The text to replace it with (must be different from old_string).",
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
        description="The directory to search in. If not specified, searches the entire filesystem.",
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
        description=("File type to search (rg --type). Common types: js, py, rust, go, java, etc."),
    )
    head_limit: int = Field(
        default=250,
        description=(
            "Limit output to first N lines/entries. Defaults to 250. Pass 0 for unlimited."
        ),
    )
    offset: int = Field(
        default=0,
        description="Skip first N lines/entries before applying head_limit.",
    )
    multiline: bool = Field(
        default=False,
        description=("Enable multiline mode where . matches newlines and patterns can span lines."),
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_file_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


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
        hunks.append(
            {
                "oldStart": old_start,
                "oldLines": old_end - old_start + 1,
                "newStart": new_start,
                "newLines": new_end - new_start + 1,
                "lines": lines,
            }
        )
    return hunks


async def _read_full_text(backend: Any, file_path: str) -> str:
    """Read full file content as raw text."""
    return str(await backend.read(file_path, limit=100_000))
