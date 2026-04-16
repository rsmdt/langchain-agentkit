"""Agents extension — blocking subagent delegation."""

from langchain_agentkit.extensions.agents.extension import AgentsExtension
from langchain_agentkit.extensions.agents.filter import (
    DEFAULT_METADATA_PREFIX,
    HideSubagentTraceExtension,
    strip_hidden_from_llm,
)
from langchain_agentkit.extensions.agents.output import (
    StrategyContext,
    SubagentOutput,
    SubagentOutputStrategy,
    full_history_strategy,
    last_message_strategy,
    resolve_output_strategy,
    trace_hidden_strategy,
)
from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.agents.types import AgentConfig

__all__ = [
    "DEFAULT_METADATA_PREFIX",
    "AgentConfig",
    "AgentsExtension",
    "HideSubagentTraceExtension",
    "StrategyContext",
    "SubagentOutput",
    "SubagentOutputStrategy",
    "create_agent_tools",
    "full_history_strategy",
    "last_message_strategy",
    "resolve_output_strategy",
    "strip_hidden_from_llm",
    "trace_hidden_strategy",
]
