"""FilesystemMiddleware — virtual filesystem tools for LangGraph agents.

Provides Claude Code-aligned file tools (Read, Write, Edit, Glob, Grep)
operating on an in-memory :class:`VirtualFilesystem`.

Usage::

    from langchain_agentkit import FilesystemMiddleware

    # Empty filesystem
    mw = FilesystemMiddleware()

    # Pre-populate from a dict
    mw = FilesystemMiddleware(files={"/config.json": '{"key": "value"}'})

    # Pre-populate from a directory
    mw = FilesystemMiddleware(files="./data")

    # Pre-configured VFS
    mw = FilesystemMiddleware(filesystem=vfs)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.tools.filesystem import create_filesystem_tools
from langchain_agentkit.vfs import VirtualFilesystem


class FilesystemMiddleware:
    """Middleware providing virtual filesystem tools.

    Tools: Read, Write, Edit, Glob, Grep — all operating on an in-memory
    :class:`VirtualFilesystem`.

    Args:
        filesystem: Optional pre-configured VirtualFilesystem. Mutually
            exclusive with ``files``.
        files: Optional initial files to load. Accepts:

            - ``dict[str, str]`` — mapping of virtual path to content
            - ``str`` or ``Path`` — real directory to load recursively
    """

    def __init__(
        self,
        filesystem: VirtualFilesystem | None = None,
        files: dict[str, str] | str | Path | None = None,
    ) -> None:
        if filesystem is not None and files is not None:
            msg = "Cannot pass both 'filesystem' and 'files'"
            raise ValueError(msg)

        if filesystem is not None:
            self.filesystem = filesystem
        else:
            self.filesystem = VirtualFilesystem()
            if files is not None:
                _load_files(self.filesystem, files)

        self._tools_cache: list[BaseTool] | None = None

    @property
    def state_schema(self) -> None:
        """No additional state keys — VFS is in-memory."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """Filesystem tools: ``[Read, Write, Edit, Glob, Grep]``."""
        if self._tools_cache is None:
            self._tools_cache = create_filesystem_tools(self.filesystem)
        return self._tools_cache

    def prompt(
        self, state: dict[str, Any], runtime: ToolRuntime | None = None,
    ) -> str | None:
        """Return filesystem prompt section, or ``None`` if no files loaded."""
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


def _load_files(
    vfs: VirtualFilesystem, files: dict[str, str] | str | Path,
) -> None:
    """Load files into a VFS from a dict or directory path."""
    if isinstance(files, dict):
        vfs.load_dict(files)
    elif isinstance(files, (str, Path)):
        vfs.load_directory(files)
    else:
        msg = f"'files' must be dict, str, or Path, got {type(files).__name__}"
        raise TypeError(msg)
