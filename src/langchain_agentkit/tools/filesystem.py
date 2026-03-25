"""Claude Code-aligned filesystem tools operating on a VirtualFilesystem.

Provides five tools: ``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``.

Usage::

    from langchain_agentkit.vfs import VirtualFilesystem
    from langchain_agentkit.tools.filesystem import create_filesystem_tools

    vfs = VirtualFilesystem()
    tools = create_filesystem_tools(vfs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.vfs import VirtualFilesystem


class _ReadInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to read.")
    offset: int = Field(default=0, description="Line number to start reading from (0-indexed).")
    limit: int = Field(default=2000, description="Maximum number of lines to read.")


class _WriteInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to write.")
    content: str = Field(description="Content to write to the file.")


class _EditInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to edit.")
    old_string: str = Field(description="Exact string to find and replace.")
    new_string: str = Field(description="Replacement string.")
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences. If False, old_string must be unique in the file.",
    )


class _GlobInput(BaseModel):
    pattern: str = Field(description='Glob pattern to match files (e.g., "/skills/**/*.md").')


class _GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression pattern to search for.")
    path: str | None = Field(default=None, description="Directory to restrict search to.")
    glob: str | None = Field(default=None, description="Glob pattern to filter files.")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")


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


def _build_read_tool(vfs: VirtualFilesystem) -> BaseTool:
    def read(file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with line numbers."""
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
            "Read a file from the virtual filesystem with line numbers. "
            "Use offset and limit for large files."
        ),
        args_schema=_ReadInput,
        handle_tool_error=True,
    )


def _build_write_tool(vfs: VirtualFilesystem) -> BaseTool:
    def write(file_path: str, content: str) -> str:
        """Write content to a file, creating or overwriting."""
        vfs.write(file_path, content)
        return f"Wrote {len(content)} characters to {file_path}"

    return StructuredTool.from_function(
        func=write,
        name="Write",
        description=(
            "Write content to a file in the virtual filesystem, creating or overwriting it."
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
        """Replace exact string occurrences in a file."""
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
            "Replace exact string occurrences in a file. "
            "old_string must be unique unless replace_all is True."
        ),
        args_schema=_EditInput,
        handle_tool_error=True,
    )


def _build_glob_tool(vfs: VirtualFilesystem) -> BaseTool:
    def glob(pattern: str) -> str:
        """Find files matching a glob pattern."""
        matches = vfs.glob(pattern)
        if not matches:
            return "No files matched."
        return "\n".join(matches)

    return StructuredTool.from_function(
        func=glob,
        name="Glob",
        description=('Find files matching a glob pattern (e.g., "/skills/**/*.md").'),
        args_schema=_GlobInput,
        handle_tool_error=True,
    )


def _build_grep_tool(vfs: VirtualFilesystem) -> BaseTool:
    def grep(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
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
        lines = [f"{r['path']}:{r['line']}: {r['text']}" for r in results]
        if len(lines) > 200:
            lines = lines[:200]
            lines.append(f"... ({len(results) - 200} more matches)")
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=grep,
        name="Grep",
        description=(
            "Search file contents for a regex pattern. "
            "Returns matching lines with file:line references."
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
