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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the agent with input and optional config."""
        ...

    async def astream(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the agent's output."""
        ...


class CompiledAgent:
    """Wraps a compiled StateGraph as an AgentLike.

    Extracts ``name`` and ``description`` from ``name`` and
    ``description`` metadata on the graph.

    Args:
        graph: A compiled StateGraph (or any object with ``ainvoke``/``astream``).
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self._name: str = getattr(graph, "name", "agent")
        self._description: str = getattr(graph, "description", "")

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
        return await self._graph.ainvoke(input, config)  # type: ignore[no-any-return]

    async def astream(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate to the underlying graph's astream."""
        async for chunk in self._graph.astream(input, config):
            yield chunk


class TeamAgent:
    """Wraps a lead agent + teammates as a single AgentLike.

    Implements the SocietyOfMindAgent pattern (AutoGen). On ``ainvoke``,
    the lead agent is invoked with the input. The teammates are available
    to the lead for delegation/coordination.

    The inner team runs to completion, then the lead's result is returned
    as the ``TeamAgent``'s response.

    Args:
        lead: The lead agent (AgentLike).
        teammates: List of AgentLike teammates.

    Example::

        research_team = TeamAgent(
            lead=researcher,
            teammates=[analyst, writer],
        )

        # Use as a single AgentLike — delegates internally
        result = await research_team.ainvoke({"messages": ["research X"]})
    """

    def __init__(self, lead: AgentLike, teammates: list[AgentLike]) -> None:
        if lead is None:
            raise ValueError("TeamAgent requires a lead agent")
        if not teammates:
            raise ValueError("TeamAgent requires at least one teammate")

        self._lead = lead
        self._teammates = list(teammates)

    @property
    def name(self) -> str:
        return self._lead.name

    @property
    def description(self) -> str:
        return self._lead.description

    @property
    def lead(self) -> AgentLike:
        """The lead agent."""
        return self._lead

    @property
    def teammates(self) -> list[AgentLike]:
        """The list of teammates."""
        return list(self._teammates)

    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run the lead agent to completion with the given input.

        The lead agent is responsible for coordinating with teammates
        (via AgentsExtension or TeamExtension tools). The TeamAgent
        delegates the full task to the lead and returns its result.
        """
        return await self._lead.ainvoke(input, config)

    async def astream(
        self, input: dict[str, Any], config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream the lead agent's output."""
        async for chunk in self._lead.astream(input, config):  # type: ignore[attr-defined]
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
    return CompiledAgent(target)  # type: ignore[return-value]
