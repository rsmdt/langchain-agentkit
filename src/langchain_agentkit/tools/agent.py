"""Backward-compat shim. Import from langchain_agentkit.extensions.agents.tools instead."""

from langchain_agentkit.extensions.agents.tools import *  # noqa: F401, F403
from langchain_agentkit.extensions.agents.tools import create_agent_tools as create_agent_tools
