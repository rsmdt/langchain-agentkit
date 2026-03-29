"""Composability protocol and adapters for agentkit.

``AgentLike`` is the protocol that both agents and teams implement,
enabling fractal nesting — a team member can be a team.

``CompiledAgent`` wraps a compiled ``StateGraph`` as an ``AgentLike``.

``wrap_if_needed`` auto-wraps raw StateGraphs for backward compatibility.

Usage::

    from langchain_agentkit.composability import AgentLike, CompiledAgent

    # Raw graph → AgentLike
    agent = CompiledAgent(my_graph.compile())

    # Auto-wrap
    agent = wrap_if_needed(my_graph)  # CompiledAgent if not already AgentLike
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class AgentLike(Protocol):
    """Protocol for composable agents and teams.

    Uses LangChain vocabulary (``ainvoke``/``astream``). Both solo agents
    and teams implement this, enabling fractal nesting.
    """

    @property
    def name(self) -> str:
        """Agent or team name."""
        ...

    @property
    def description(self) -> str:
        """Agent or team description."""
        ...

    async def ainvoke(self, input: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke the agent with input and optional config."""
        ...

    async def astream(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the agent's output."""
        ...


class CompiledAgent:
    """Wraps a compiled StateGraph as an AgentLike.

    Extracts ``name`` and ``description`` from ``agentkit_name`` and
    ``agentkit_description`` metadata on the graph.

    Args:
        graph: A compiled StateGraph (or any object with ``ainvoke``/``astream``).
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self._name: str = getattr(graph, "agentkit_name", "agent")
        self._description: str = getattr(graph, "agentkit_description", "")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def graph(self) -> Any:
        """The underlying compiled graph."""
        return self._graph

    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Delegate to the underlying graph's ainvoke."""
        return await self._graph.ainvoke(input, config)

    async def astream(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate to the underlying graph's astream."""
        async for chunk in self._graph.astream(input, config):
            yield chunk


def wrap_if_needed(target: Any) -> AgentLike:
    """Wrap a target as AgentLike if it isn't already.

    If ``target`` already satisfies ``AgentLike``, returns it unchanged.
    Otherwise wraps it in a ``CompiledAgent``.

    Args:
        target: An AgentLike instance, compiled StateGraph, or any
            object with ``ainvoke``/``astream``.

    Returns:
        An AgentLike instance.
    """
    if isinstance(target, AgentLike):
        return target
    return CompiledAgent(target)
