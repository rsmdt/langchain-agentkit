"""Middleware protocol and implementations for langchain-agentkit.

The ``Middleware`` protocol defines the contract. Implementations live
in submodules: ``filesystem``, ``skills``, ``tasks``, ``hitl``, ``web_search``.

Re-exports::

    from langchain_agentkit.middleware import Middleware
    from langchain_agentkit.middleware import FilesystemMiddleware
    from langchain_agentkit.middleware import SkillsMiddleware
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


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


from langchain_agentkit.middleware.agents import AgentMiddleware as AgentMiddleware
from langchain_agentkit.middleware.filesystem import FilesystemMiddleware as FilesystemMiddleware
from langchain_agentkit.middleware.hitl import HITLMiddleware as HITLMiddleware
from langchain_agentkit.middleware.skills import SkillsMiddleware as SkillsMiddleware
from langchain_agentkit.middleware.tasks import TasksMiddleware as TasksMiddleware
from langchain_agentkit.middleware.teams import AgentTeamMiddleware as AgentTeamMiddleware
from langchain_agentkit.middleware.web_search import (
    DuckDuckGoSearchProvider as DuckDuckGoSearchProvider,
)
from langchain_agentkit.middleware.web_search import (
    QwantSearchProvider as QwantSearchProvider,
)
from langchain_agentkit.middleware.web_search import (
    WebSearchMiddleware as WebSearchMiddleware,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime


class Middleware(Protocol):
    """Protocol for middleware that contributes tools, prompts, and state to an agent."""

    @property
    def tools(self) -> list[BaseTool]:
        """Tools this middleware provides to the LLM."""
        ...

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str | None:
        """Prompt section to inject into the system prompt.

        Called on every LLM invocation. Return None to skip injection.
        """
        ...

    @property
    def state_schema(self) -> type | None:
        """TypedDict mixin for this middleware's state requirements.

        Return a TypedDict class to add keys to the graph state, or
        ``None`` if no additional state keys are needed.

        Example::

            from langchain_agentkit.state import TasksState

            @property
            def state_schema(self) -> type:
                return TasksState
        """
        ...

    def dependencies(self) -> list[Any]:
        """Optional: middleware this one depends on. Auto-added if missing.

        Return instances of required middleware. The dependency resolver
        deduplicates by type — if the user already added the middleware,
        it won't be duplicated.

        Default: no dependencies.
        """
        return []

__all__ = [
    "AgentMiddleware",
    "AgentTeamMiddleware",
    "DuckDuckGoSearchProvider",
    "FilesystemMiddleware",
    "HITLMiddleware",
    "Middleware",
    "QwantSearchProvider",
    "SkillsMiddleware",
    "TasksMiddleware",
    "WebSearchMiddleware",
]
