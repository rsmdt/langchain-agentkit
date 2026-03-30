"""FilesystemExtension — file tools for LangGraph agents.

Provides Claude Code-aligned file tools (Read, Write, Edit, MultiEdit,
Glob, Grep, LS) operating on a :class:`BackendProtocol` backend.

Usage::

    from langchain_agentkit import FilesystemExtension

    # Default: OS filesystem rooted at current directory
    ext = FilesystemExtension()

    # Explicit root directory
    ext = FilesystemExtension(root="./workspace")

    # Custom backend (e.g. Daytona sandbox)
    ext = FilesystemExtension(backend=my_backend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from langchain_agentkit.backend import BackendProtocol, OSBackend
from langchain_agentkit.extension import Extension
from langchain_agentkit.tools.filesystem import create_filesystem_tools

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime


class FilesystemExtension(Extension):
    """Extension providing filesystem tools.

    Tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS.

    Args:
        backend: A :class:`BackendProtocol` implementation. When provided,
            ``root`` is ignored.
        root: Root directory for the default OS filesystem backend.
            Defaults to the current working directory.
    """

    def __init__(
        self,
        backend: BackendProtocol | None = None,
        root: str | Path = ".",
    ) -> None:
        self._backend = backend or OSBackend(str(root))
        self._tools_cache: list[BaseTool] | None = None

    @property
    def backend(self) -> BackendProtocol:
        """The backend this extension operates on."""
        return self._backend

    @property
    def state_schema(self) -> None:
        """No additional state keys."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """Filesystem tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS."""
        if self._tools_cache is None:
            tools = create_filesystem_tools(self._backend)
            tools.append(_build_ls_tool(self._backend))
            tools.append(_build_multi_edit_tool(self._backend))
            self._tools_cache = tools
        return self._tools_cache

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> str | None:
        """Return filesystem prompt section."""
        root = ""
        if isinstance(self._backend, OSBackend):
            root = f" rooted at `{self._backend._root}`"
        return (
            f"## Filesystem\n\n"
            f"You have access to a filesystem{root}.\n\n"
            f"Use the Read, Write, Edit, Glob, and Grep tools to "
            f"interact with files."
        )


# --- Tool builders ---


# --- Input schemas for tool builders (module-level for type resolution) ---


class _LSInput(BaseModel):
    path: str = Field(
        default="/",
        description="The directory path to list.",
    )


class _EditOperation(BaseModel):
    old_string: str = Field(description="The text to replace.")
    new_string: str = Field(description="The replacement text.")


class _MultiEditInput(BaseModel):
    file_path: str = Field(description="The file to edit.")
    edits: list[_EditOperation] = Field(
        description="List of edit operations.",
    )


def _build_ls_tool(backend: BackendProtocol) -> BaseTool:
    """Build the LS tool for directory listing."""

    def _ls(path: str = "/") -> str:
        entries = backend.ls(path)
        if not entries:
            return f"Directory '{path}' is empty or does not exist."
        lines = []
        for entry in entries:
            indicator = "/" if entry.get("is_dir", False) else ""
            size = entry.get("size", 0)
            lines.append(f"{entry['path']}{indicator}  ({size} bytes)")
        return "\n".join(lines)

    return StructuredTool(
        name="LS",
        description="List files and directories at the given path.",
        func=_ls,
        args_schema=_LSInput,
    )


def _build_multi_edit_tool(backend: BackendProtocol) -> BaseTool:
    """Build the MultiEdit tool for batch edits."""

    def _multi_edit(
        file_path: str, edits: list[_EditOperation],
    ) -> str:
        total = 0
        for edit in edits:
            result = backend.edit(
                file_path, edit.old_string, edit.new_string,
            )
            total += result.get("replacements", 0)
        return (
            f"Applied {total} replacement(s) across "
            f"{len(edits)} edit(s) in {file_path}."
        )

    return StructuredTool(
        name="MultiEdit",
        description=(
            "Apply multiple find-and-replace edits to a single "
            "file in one operation."
        ),
        func=_multi_edit,
        args_schema=_MultiEditInput,
    )
