"""FilesystemExtension — file and execution tools for LangGraph agents.

Provides Claude Code-aligned file tools (Read, Write, Edit, MultiEdit,
Glob, Grep, LS) operating on a :class:`BackendProtocol` backend.
Optionally includes an Execute tool for shell commands when the backend
implements :class:`SandboxProtocol`.

Usage::

    from langchain_agentkit import FilesystemExtension, MemoryBackend, LocalBackend

    # In-memory (default, backward-compatible)
    ext = FilesystemExtension()

    # Local filesystem with path sandboxing
    ext = FilesystemExtension(backend=LocalBackend("./workspace"))

    # With shell execution enabled
    ext = FilesystemExtension(backend=my_sandbox, include_execute=True)

    # Legacy: pre-configured VFS
    ext = FilesystemExtension(filesystem=vfs)

    # Legacy: pre-populate from dict or directory
    ext = FilesystemExtension(files={"/config.json": '{"key": "value"}'})
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.backend import BackendProtocol, MemoryBackend, SandboxProtocol
from langchain_agentkit.tools.filesystem import create_backend_tools, create_filesystem_tools
from langchain_agentkit.vfs import VirtualFilesystem


class FilesystemExtension(Extension):
    """Extension providing filesystem and optional execution tools.

    Tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS — plus Execute
    when enabled and the backend supports it.

    Args:
        backend: A :class:`BackendProtocol` implementation. When provided,
            ``filesystem`` and ``files`` are ignored.
        include_execute: If True and the backend implements
            :class:`SandboxProtocol`, include the Execute tool.
        filesystem: Legacy — pre-configured VirtualFilesystem.
        files: Legacy — initial files to load (dict, str path, or Path).
    """

    def __init__(
        self,
        backend: BackendProtocol | None = None,
        include_execute: bool = False,
        filesystem: VirtualFilesystem | None = None,
        files: dict[str, str] | str | Path | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
            # Extract VFS from MemoryBackend for backward compat
            if isinstance(backend, MemoryBackend):
                self.filesystem = backend._vfs
            else:
                self.filesystem = VirtualFilesystem()  # Empty placeholder
        elif filesystem is not None:
            if files is not None:
                msg = "Cannot pass both 'filesystem' and 'files'"
                raise ValueError(msg)
            self.filesystem = filesystem
            self._backend = MemoryBackend()
            self._backend._vfs = filesystem
        else:
            self.filesystem = VirtualFilesystem()
            if files is not None:
                _load_files(self.filesystem, files)
            self._backend = MemoryBackend()
            self._backend._vfs = self.filesystem

        self._include_execute = include_execute
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
        """Filesystem tools: Read, Write, Edit, MultiEdit, Glob, Grep, LS.

        Plus Execute if ``include_execute=True`` and backend is a SandboxProtocol.

        When a non-MemoryBackend is provided, all core tools (Read, Write, Edit,
        Glob, Grep) use the BackendProtocol directly. When using MemoryBackend
        (or legacy VFS), they use the VirtualFilesystem for backward compat.
        """
        if self._tools_cache is None:
            # Core tools — use backend-aware builders when backend is not MemoryBackend
            if isinstance(self._backend, MemoryBackend):
                tools = create_filesystem_tools(self.filesystem)
            else:
                tools = create_backend_tools(self._backend)

            # Add LS tool (always uses backend)
            tools.append(_build_ls_tool(self._backend))

            # Add MultiEdit tool (always uses backend)
            tools.append(_build_multi_edit_tool(self._backend))

            # Conditionally add Execute
            if self._include_execute and isinstance(self._backend, SandboxProtocol):
                tools.append(_build_execute_tool(self._backend))

            self._tools_cache = tools
        return self._tools_cache

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> str | None:
        """Return filesystem prompt section, or None if no files loaded."""
        file_count = len(self.filesystem)
        if file_count == 0:
            return None

        dirs = set()
        for path in self.filesystem.files:
            parts = path.strip("/").split("/")
            if len(parts) > 1:
                dirs.add("/" + parts[0] + "/")

        dir_listing = ", ".join(sorted(dirs)) if dirs else "/"
        return (
            f"## Virtual Filesystem\n\n"
            f"You have access to a virtual filesystem with {file_count} file(s) "
            f"in: {dir_listing}\n\n"
            f"Use the Read, Write, Edit, Glob, and Grep tools to "
            f"interact with files."
        )


# --- New tool builders ---

def _build_ls_tool(backend: BackendProtocol) -> BaseTool:
    """Build the LS tool for directory listing."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _LSInput(BaseModel):
        path: str = Field(
            default="/",
            description="The directory path to list.",
        )

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
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _EditOperation(BaseModel):
        old_string: str = Field(description="The text to replace.")
        new_string: str = Field(description="The replacement text.")

    class _MultiEditInput(BaseModel):
        file_path: str = Field(description="The file to edit.")
        edits: list[_EditOperation] = Field(description="List of edit operations.")

    def _multi_edit(file_path: str, edits: list[_EditOperation]) -> str:
        total = 0
        for edit in edits:
            result = backend.edit(file_path, edit.old_string, edit.new_string)
            total += result.get("replacements", 0)
        return f"Applied {total} replacement(s) across {len(edits)} edit(s) in {file_path}."

    return StructuredTool(
        name="MultiEdit",
        description="Apply multiple find-and-replace edits to a single file in one operation.",
        func=_multi_edit,
        args_schema=_MultiEditInput,
    )


def _build_execute_tool(backend: SandboxProtocol) -> BaseTool:
    """Build the Execute tool for shell command execution."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class _ExecuteInput(BaseModel):
        command: str = Field(description="The shell command to execute.")
        timeout: int | None = Field(
            default=None,
            description="Optional timeout in seconds.",
        )

    def _execute(command: str, timeout: int | None = None) -> str:
        result = backend.execute(command, timeout=timeout)
        output = result.get("output", "")
        exit_code = result.get("exit_code", -1)
        truncated = result.get("truncated", False)
        parts = [f"Exit code: {exit_code}"]
        if output:
            parts.append(output)
        if truncated:
            parts.append("[Output truncated]")
        return "\n".join(parts)

    return StructuredTool(
        name="Execute",
        description="Execute a shell command in the sandbox environment.",
        func=_execute,
        args_schema=_ExecuteInput,
    )


def _load_files(
    vfs: VirtualFilesystem,
    files: dict[str, str] | str | Path,
) -> None:
    """Load files into a VFS from a dict or directory path."""
    if isinstance(files, dict):
        vfs.load_dict(files)
    elif isinstance(files, (str, Path)):
        vfs.load_directory(files)
    else:
        msg = f"'files' must be dict, str, or Path, got {type(files).__name__}"
        raise TypeError(msg)
