"""Claude Code-aligned filesystem tools operating on a VirtualFilesystem.

Provides five tools: ``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``.
Parameters and descriptions match the Claude Code tool API.

Usage::

    from langchain_agentkit.vfs import VirtualFilesystem
    from langchain_agentkit.tools.filesystem import create_filesystem_tools

    vfs = VirtualFilesystem()
    tools = create_filesystem_tools(vfs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.vfs import VirtualFilesystem


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


def _format_with_line_numbers(content: str, offset: int, limit: int) -> str:
    """Format content with line numbers, applying offset and limit."""
    lines = content.splitlines()
    total = len(lines)
    start = min(offset, total)
    end = min(start + limit, total)
    selected = lines[start:end]

    width = len(str(end))
    numbered = [f"{i:>{width}}\t{line}" for i, line in enumerate(selected, start + 1)]

    result = "\n".join(numbered)
    if end < total:
        result += f"\n... ({total - end} more lines)"
    return result


def _format_grep_content(
    results: list[dict[str, Any]],
    all_lines: dict[str, list[str]],
    context: int | None,
) -> list[str]:
    """Format grep results as content lines with optional context."""
    output: list[str] = []
    for r in results:
        file_path = r["path"]
        line_num = r["line"]
        if context and file_path in all_lines:
            lines = all_lines[file_path]
            start = max(0, line_num - 1 - context)
            end = min(len(lines), line_num + context)
            for i in range(start, end):
                prefix = ":" if i == line_num - 1 else "-"
                output.append(f"{file_path}{prefix}{i + 1}: {lines[i]}")
            output.append("--")
        else:
            output.append(f"{file_path}:{line_num}: {r['text']}")
    return output


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def _build_read_tool(vfs: VirtualFilesystem) -> BaseTool:
    def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file from the virtual filesystem."""
        content = vfs.read(file_path)
        if content is None:
            raise ToolException(f"File not found: {file_path}")
        if not content:
            return "(empty file)"
        return _format_with_line_numbers(content, offset, limit)

    return StructuredTool.from_function(
        func=read,
        name="Read",
        description=(
            "Read a file from the virtual filesystem. "
            "Results are returned with line numbers starting at 1. "
            "Use offset and limit for large files."
        ),
        args_schema=_ReadInput,
        handle_tool_error=True,
    )


def _build_write_tool(vfs: VirtualFilesystem) -> BaseTool:
    def write(file_path: str, content: str) -> str:
        """Write content to a file."""
        vfs.write(file_path, content)
        return f"Wrote {len(content)} characters to {file_path}"

    return StructuredTool.from_function(
        func=write,
        name="Write",
        description=(
            "Write content to a file, creating or overwriting it. "
            "Prefer the Edit tool for modifying existing files."
        ),
        args_schema=_WriteInput,
        handle_tool_error=True,
    )


def _build_edit_tool(vfs: VirtualFilesystem) -> BaseTool:
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Perform exact string replacement in a file."""
        try:
            count = vfs.edit(
                file_path,
                old_string,
                new_string,
                replace_all=replace_all,
            )
        except FileNotFoundError as err:
            raise ToolException(f"File not found: {file_path}") from err
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        return f"Replaced {count} occurrence(s) in {file_path}"

    return StructuredTool.from_function(
        func=edit,
        name="Edit",
        description=(
            "Perform exact string replacements in a file. "
            "The edit will FAIL if old_string is not unique in the file. "
            "Provide more surrounding context to make it unique, "
            "or use replace_all to change every instance."
        ),
        args_schema=_EditInput,
        handle_tool_error=True,
    )


def _build_glob_tool(vfs: VirtualFilesystem) -> BaseTool:
    def glob(pattern: str, path: str | None = None) -> str:
        """Find files matching a glob pattern."""
        if path and not pattern.startswith("/"):
            pattern = f"{path.rstrip('/')}/{pattern}"
        matches = vfs.glob(pattern)
        if not matches:
            return "No files matched."
        return "\n".join(matches)

    return StructuredTool.from_function(
        func=glob,
        name="Glob",
        description=(
            "Fast file pattern matching tool. "
            'Supports glob patterns like "**/*.py" or "/skills/**/*.md". '
            "Returns matching file paths sorted alphabetically."
        ),
        args_schema=_GlobInput,
        handle_tool_error=True,
    )


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


def _grep_content(
    vfs: VirtualFilesystem,
    results: list[dict[str, Any]],
    context: int | None,
    head_limit: int,
) -> str:
    all_lines: dict[str, list[str]] = {}
    if context:
        for r in results:
            fp = r["path"]
            if fp not in all_lines:
                file_content = vfs.read(fp)
                if file_content:
                    all_lines[fp] = file_content.splitlines()

    lines = _format_grep_content(results, all_lines, context)
    if head_limit:
        lines = lines[:head_limit]
        remaining = len(results) - head_limit
        if remaining > 0:
            lines.append(f"... ({remaining} more matches)")
    elif len(lines) > 500:
        total = len(lines)
        lines = lines[:500]
        lines.append(f"... ({total - 500} more lines)")
    return "\n".join(lines)


def _build_grep_tool(vfs: VirtualFilesystem) -> BaseTool:
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
        results = vfs.grep(
            pattern,
            path=path,
            glob_filter=glob,
            ignore_case=ignore_case,
        )
        if not results:
            return "No matches found."

        if output_mode == "files_with_matches":
            return _grep_files_with_matches(results, head_limit)
        elif output_mode == "count":
            return _grep_count(results, head_limit)
        else:
            return _grep_content(vfs, results, context, head_limit)

    return StructuredTool.from_function(
        func=grep,
        name="Grep",
        description=(
            "Search file contents for a regex pattern. "
            "Supports full regex syntax. "
            "Output modes: "
            '"content" shows matching lines, '
            '"files_with_matches" shows only file paths (default), '
            '"count" shows match counts per file.'
        ),
        args_schema=_GrepInput,
        handle_tool_error=True,
    )


def create_filesystem_tools(vfs: VirtualFilesystem) -> list[BaseTool]:
    """Create Claude Code-aligned filesystem tools for a VirtualFilesystem.

    Returns five tools: ``[Read, Write, Edit, Glob, Grep]``.

    Args:
        vfs: The virtual filesystem instance these tools operate on.
    """
    return [
        _build_read_tool(vfs),
        _build_write_tool(vfs),
        _build_edit_tool(vfs),
        _build_glob_tool(vfs),
        _build_grep_tool(vfs),
    ]
