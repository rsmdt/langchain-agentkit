"""Agents extension — blocking subagent delegation."""

from langchain_agentkit.extensions.agents.extension import AgentExtension
from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.agents.types import AgentConfig

__all__ = ["AgentConfig", "AgentExtension", "create_agent_tools"]
