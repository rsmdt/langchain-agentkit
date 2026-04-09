"""Agent configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Agent definition — used by both programmatic and file-based agents.

    At delegation time, these fields are resolved:
    - ``model``: resolved via ``model_resolver`` if string, or used as-is
    - ``tools``: filtered from parent's available tools by name
    - ``skills``: resolved by name, content concatenated into prompt
    - ``max_turns``: used as recursion limit on the compiled graph
    """

    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    model: str | None = None
    max_turns: int | None = None
    skills: list[str] | None = None


class _AgentConfigProxy:
    """Proxy that makes an AgentConfig look like an agent to the roster.

    Has ``name`` and ``description``. The delegation tool detects the
    ``_agent_config`` attribute and routes to the definition-based
    delegation path before any ``tools_inherit`` check is reached.
    """

    def __init__(self, definition: AgentConfig) -> None:
        self.name = definition.name
        self.description = definition.description
        self._agent_config = definition


def _wrap_agents(agents: list[Any]) -> list[Any]:
    """Wrap AgentConfig instances in proxies, pass everything else through."""
    result = []
    for a in agents:
        if isinstance(a, AgentConfig):
            result.append(_AgentConfigProxy(a))
        else:
            result.append(a)
    return result
