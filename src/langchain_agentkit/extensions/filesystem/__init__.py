"""Filesystem extension — Claude Code-aligned file tools."""

from langchain_agentkit.extensions.filesystem.extension import FilesystemExtension
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools

__all__ = ["FilesystemExtension", "create_filesystem_tools"]
