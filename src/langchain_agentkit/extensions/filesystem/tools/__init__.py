"""Filesystem tools operating on a BackendProtocol.

Provides five tools: ``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep``.
Parameters and descriptions match the Claude Code tool API.

Usage::

    from langchain_agentkit.backends.os import OSBackend
    from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools

    backend = OSBackend("./workspace")
    tools = create_filesystem_tools(backend)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from langchain_agentkit.extensions.filesystem.tools.edit import _build_edit
from langchain_agentkit.extensions.filesystem.tools.glob import _build_glob
from langchain_agentkit.extensions.filesystem.tools.grep import _build_grep
from langchain_agentkit.extensions.filesystem.tools.read import _build_read
from langchain_agentkit.extensions.filesystem.tools.write import _build_write


def create_filesystem_tools(backend: Any) -> list[BaseTool]:
    """Create filesystem tools backed by a BackendProtocol.

    Returns five tools: ``[Read, Write, Edit, Glob, Grep]``,
    all operating on the given backend. Each call creates an
    isolated read-state cache for file-unchanged dedup.

    Args:
        backend: A BackendProtocol implementation.
    """
    # Per-backend read-state cache (not module-level) to prevent
    # cross-instance cache pollution in multi-agent scenarios.
    read_state: dict[str, str] = {}
    return [
        _build_read(backend, read_state),
        _build_write(backend),
        _build_edit(backend),
        _build_glob(backend),
        _build_grep(backend),
    ]
