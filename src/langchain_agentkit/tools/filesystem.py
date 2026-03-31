"""Backward-compat shim. Import from langchain_agentkit.extensions.filesystem.tools instead."""

from langchain_agentkit.extensions.filesystem.tools import *  # noqa: F401, F403
from langchain_agentkit.extensions.filesystem.tools import (
    create_filesystem_tools as create_filesystem_tools,
)
