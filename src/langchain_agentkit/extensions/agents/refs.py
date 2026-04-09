"""Shared agent reference types and roster helpers.

These are the public contracts that both ``AgentExtension`` and
``TeamExtension`` depend on. Keeping them in a neutral module avoids
cross-extension private imports and makes the shape of a roster entry
explicit.
"""

from __future__ import annotations

import logging
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
