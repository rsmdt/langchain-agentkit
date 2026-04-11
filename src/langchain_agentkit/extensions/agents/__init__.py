"""Agents extension — blocking subagent delegation."""

from langchain_agentkit.extensions.agents.extension import AgentsExtension
from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.agents.types import AgentConfig

__all__ = ["AgentConfig", "AgentsExtension", "create_agent_tools"]
