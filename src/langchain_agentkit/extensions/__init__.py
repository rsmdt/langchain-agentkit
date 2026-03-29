"""Extension implementations for langchain-agentkit.

Re-exports::

    from langchain_agentkit.extensions import AgentExtension
    from langchain_agentkit.extensions import TasksExtension
    from langchain_agentkit.extensions import FilesystemExtension
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


def validate_agent_list(agents: list[Any]) -> dict[str, Any]:
    """Validate agent list and return agents_by_name dict.

    Raises ValueError if: empty list, missing agentkit_name, duplicate names.
    """
    if not agents:
        raise ValueError("agents list cannot be empty")
    names = [getattr(g, "agentkit_name", None) for g in agents]
    if any(n is None for n in names):
        raise ValueError("All agents must have agentkit_name (use the agent metaclass)")
    if len(set(names)) != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate agent names: {set(dupes)}")
    return {g.agentkit_name: g for g in agents}


def resolve_agent(agent_name: str, agents_by_name: dict[str, Any]) -> Any:
    """Look up an agent by name, raising a descriptive error if not found.

    Returns the agent graph on success. Raises ``ToolException`` with
    available agent names on failure.
    """
    from langchain_core.tools import ToolException

    if agent_name not in agents_by_name:
        available = ", ".join(sorted(agents_by_name.keys()))
        raise ToolException(
            f"Agent '{agent_name}' not found. Available agents: {available}"
        )
    return agents_by_name[agent_name]


from langchain_agentkit.extensions.agents import AgentExtension as AgentExtension
from langchain_agentkit.extensions.filesystem import FilesystemExtension as FilesystemExtension
from langchain_agentkit.extensions.hitl import HITLExtension as HITLExtension
from langchain_agentkit.extensions.skills import SkillsExtension as SkillsExtension
from langchain_agentkit.extensions.tasks import TasksExtension as TasksExtension
from langchain_agentkit.extensions.teams import TeamExtension as TeamExtension
from langchain_agentkit.extensions.web_search import (
    DuckDuckGoSearchProvider as DuckDuckGoSearchProvider,
)
from langchain_agentkit.extensions.web_search import (
    QwantSearchProvider as QwantSearchProvider,
)
from langchain_agentkit.extensions.web_search import (
    WebSearchExtension as WebSearchExtension,
)

__all__ = [
    "AgentExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HITLExtension",
    "QwantSearchProvider",
    "SkillsExtension",
    "TasksExtension",
    "TeamExtension",
    "WebSearchExtension",
]
