"""FilesystemMiddleware — virtual filesystem tools for LangGraph agents.

Provides Claude Code-aligned file tools (Read, Write, Edit, Glob, Grep)
operating on an in-memory :class:`VirtualFilesystem`.

Usage::

    from langchain_agentkit import FilesystemMiddleware

    mw = FilesystemMiddleware()
    mw.tools   # [Read, Write, Edit, Glob, Grep]

Pre-populate with files::

    from langchain_agentkit.virtual_filesystem import VirtualFilesystem

    vfs = VirtualFilesystem()
    vfs.write("/data/config.json", '{"key": "value"}')
    mw = FilesystemMiddleware(vfs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.filesystem_tools import create_filesystem_tools
from langchain_agentkit.virtual_filesystem import VirtualFilesystem


class FilesystemMiddleware:
    """Middleware providing virtual filesystem tools.

    Tools: Read, Write, Edit, Glob, Grep — all operating on an in-memory
    :class:`VirtualFilesystem`.

    Args:
        filesystem: Optional pre-configured filesystem. If ``None``,
            creates an empty one.
    """

    def __init__(self, filesystem: VirtualFilesystem | None = None) -> None:
        self.filesystem = filesystem or VirtualFilesystem()
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """Filesystem tools: ``[Read, Write, Edit, Glob, Grep]``."""
        if self._tools_cache is None:
            self._tools_cache = create_filesystem_tools(self.filesystem)
        return self._tools_cache

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str | None:
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
            f"Use the Read, Write, Edit, Glob, and Grep tools to interact with files."
        )
