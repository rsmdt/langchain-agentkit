"""User-facing orchestration surface.

Composes agents and kits from extensions: the ``Agent`` declarative class,
the ``AgentKit`` primitive, the ``Extension`` base, hook decorators, shared
state, prompt composition, and agent-composability protocols.
"""

from langchain_agentkit.composition.agent import Agent
from langchain_agentkit.composition.agent_kit import AgentKit, run_extension_setup
from langchain_agentkit.composition.composability import (
    AgentLike,
    CompiledAgent,
    TeamAgent,
    wrap_if_needed,
)
from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.composition.hooks import after, before, wrap
from langchain_agentkit.composition.prompts import PromptComposition
from langchain_agentkit.composition.state import AgentKitState

__all__ = [
    "Agent",
    "AgentKit",
    "AgentKitState",
    "AgentLike",
    "CompiledAgent",
    "Extension",
    "PromptComposition",
    "TeamAgent",
    "after",
    "before",
    "run_extension_setup",
    "wrap",
    "wrap_if_needed",
]
