"""Shared agent reference types and roster helpers.

These are the public contracts that both ``AgentsExtension`` and
``TeamExtension`` depend on. Keeping them in a neutral module avoids
cross-extension private imports and makes the shape of a roster entry
explicit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field


class Predefined(BaseModel):
    """Select a pre-defined agent from the roster."""

    id: str = Field(description="Agent name from the available roster.")


class Dynamic(BaseModel):
    """Create an on-the-fly reasoning agent."""

    prompt: str = Field(
        description="System prompt defining the agent's role and behavior.",
    )


def validate_agent_list(agents: list[Any]) -> dict[str, Any]:
    """Validate an agent list and return an ``agents_by_name`` dict.

    Requires every agent to have a unique, non-empty ``name`` attribute.
    """
    if not agents:
        raise ValueError("agents list cannot be empty")
    names = [getattr(g, "name", None) for g in agents]
    if any(n is None for n in names):
        raise ValueError("All agents must have a name")
    if len(set(names)) != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate agent names: {set(dupes)}")
    return {name: agent for name, agent in zip(names, agents, strict=True)}  # type: ignore[misc]


def load_agent_roster(
    agents: list[Any] | str | Path,
    backend: Any,
) -> tuple[dict[str, Any], str | None, bool]:
    """Normalize the ``agents`` constructor argument into a roster.

    Shared by ``AgentsExtension`` and ``TeamExtension``. Returns
    ``(agents_by_name, deferred_path, has_config_agents)``:

    - **list** — wrap raw entries, validate, roster is immediate.
    - **str/Path with a backend** — discovery must run against the
      backend's filesystem, so it is deferred to async ``setup()``;
      ``deferred_path`` carries the directory and the roster is empty.
    - **str/Path without a backend** — synchronous host-filesystem
      discovery.
    """
    from langchain_agentkit.extensions.agents.discovery import (
        discover_agents_from_directory,
    )
    from langchain_agentkit.extensions.agents.types import (
        _AgentConfigProxy,
        _wrap_agents,
    )

    if isinstance(agents, list):
        wrapped = _wrap_agents(agents)
        has_config = any(isinstance(a, _AgentConfigProxy) for a in wrapped)
        return validate_agent_list(wrapped), None, has_config
    if isinstance(agents, (str, Path)):
        if backend is not None:
            # Defer async discovery to setup().
            return {}, str(agents), False
        defs = discover_agents_from_directory(Path(agents))
        proxies = [_AgentConfigProxy(d) for d in defs]
        roster = validate_agent_list(proxies) if proxies else {}
        return roster, None, bool(proxies)
    raise TypeError(f"agents must be list, str, or Path, got {type(agents).__name__}")


def resolve_agent_by_name(agent_name: str, agents_by_name: dict[str, Any]) -> Any:
    """Look up an agent by name, raising ``ToolException`` if absent."""
    if agent_name not in agents_by_name:
        logging.getLogger(__name__).warning(
            "Agent '%s' not found. Registered: %s",
            agent_name,
            sorted(agents_by_name.keys()),
        )
        raise ToolException(
            f"Agent '{agent_name}' not found. Check the agent roster for available names."
        )
    return agents_by_name[agent_name]
