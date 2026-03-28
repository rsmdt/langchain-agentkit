"""AgentTeamMiddleware — message-driven team coordination.

Teammates run as asyncio.Tasks. The lead coordinates via tools
(SpawnTeam, AssignTask, MessageTeammate, CheckTeammates, DissolveTeam).
Communication flows through a ``TeamMessageBus`` backed by asyncio.Queue.

Usage::

    from langchain_agentkit import agent, AgentTeamMiddleware

    class project_lead(agent):
        llm = ChatOpenAI(model="gpt-4o")
        middleware = [AgentTeamMiddleware([researcher, coder])]
        prompt = "You are a project lead managing a development team."
        async def handler(state, *, llm, tools, prompt):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# TeamMessage & TeamMessageBus
# ---------------------------------------------------------------------------


@dataclass
class TeamMessage:
    """A single message passed between team members via the bus."""

    id: str
    sender: str
    receiver: str
    content: str
    timestamp: float


class TeamMessageBus:
    """asyncio.Queue-based message bus for team coordination.

    Inspired by AutoGen v0.4's SingleThreadedAgentRuntime:
    - One asyncio.Queue per registered agent
    - FIFO ordering, no locks needed
    - Cooperative concurrency at await points
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[TeamMessage]] = {}

    def register(self, agent_name: str) -> None:
        """Register an agent, creating its message queue."""
        if agent_name in self._queues:
            return  # idempotent
        self._queues[agent_name] = asyncio.Queue()

    def unregister(self, agent_name: str) -> None:
        """Remove an agent's queue."""
        self._queues.pop(agent_name, None)

    async def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
    ) -> None:
        """Send a message to a specific agent."""
        queue = self._queues.get(to_agent)
        if queue is None:
            raise ValueError(f"Agent '{to_agent}' is not registered on the bus.")
        message = TeamMessage(
            id=str(uuid.uuid4()),
            sender=from_agent,
            receiver=to_agent,
            content=content,
            timestamp=time.time(),
        )
        await queue.put(message)

    async def broadcast(self, from_agent: str, content: str) -> None:
        """Send a message to all registered agents except the sender."""
        for name in self._queues:
            if name != from_agent:
                await self.send(from_agent, name, content)

    async def receive(
        self,
        agent_name: str,
        timeout: float = 5.0,
    ) -> TeamMessage | None:
        """Receive next message for an agent, with timeout.

        Returns ``None`` if no message arrives within timeout.
        """
        queue = self._queues.get(agent_name)
        if queue is None:
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def pending_count(self, agent_name: str) -> int:
        """Return number of unread messages for an agent."""
        queue = self._queues.get(agent_name)
        if queue is None:
            return 0
        return queue.qsize()


# ---------------------------------------------------------------------------
# Teammate execution loop
# ---------------------------------------------------------------------------


async def _teammate_loop(
    member_name: str,
    compiled_graph: Any,
    message_bus: TeamMessageBus,
) -> str:
    """Event loop for a single teammate.

    Blocks on ``receive()``, processes messages via the compiled graph,
    and sends results back to the sender. Exits on ``"__shutdown__"`` signal.
    """
    while True:
        msg = await message_bus.receive(member_name, timeout=30.0)
        if msg is None:
            continue  # idle, waiting for work
        if msg.content == "__shutdown__":
            return "shutdown"

        # Execute the full ReAct loop
        try:
            result = await compiled_graph.ainvoke({
                "messages": [HumanMessage(content=msg.content)],
                "sender": member_name,
            })
            final = (
                result["messages"][-1].content
                if result.get("messages")
                else "No response"
            )
        except Exception as exc:
            final = f"Error during execution: {exc}"

        # Report back to whoever sent the message
        await message_bus.send(member_name, msg.sender, final)


# ---------------------------------------------------------------------------
# ActiveTeam dataclass
# ---------------------------------------------------------------------------


@dataclass
class ActiveTeam:
    """Runtime state for an active team — lives on the middleware instance."""

    name: str
    bus: TeamMessageBus
    members: dict[str, asyncio.Task[str]]  # name → asyncio.Task
    member_types: dict[str, str]  # name → agent_type
    iteration_count: int = field(default=0)


# ---------------------------------------------------------------------------
# AgentTeamMiddleware
# ---------------------------------------------------------------------------


class AgentTeamMiddleware:
    """Middleware providing message-driven team coordination.

    Teammates run as asyncio.Tasks. The lead reacts to teammate messages
    (not polls). Modeled after Claude Code's Agent Team pattern with
    AutoGen-inspired message dispatch primitives.

    Args:
        agents: List of StateGraph objects with ``.agentkit_name`` and
            ``.agentkit_description`` attributes (from the agent metaclass).
        ephemeral: Enable ephemeral team members (reserved for future use).
        max_team_size: Maximum number of team members allowed.
        router_timeout: Seconds to wait for messages in the Router Node.
        max_iterations: Safety limit for Router Node re-invocations.

    Example::

        from langchain_agentkit import agent, AgentTeamMiddleware

        class lead(agent):
            llm = ChatOpenAI(model="gpt-4o")
            middleware = [AgentTeamMiddleware([researcher, coder])]
            prompt = "You are a project lead."
            async def handler(state, *, llm, tools, prompt):
                response = await llm.ainvoke(state["messages"])
                return {"messages": [response]}
    """

    def __init__(
        self,
        agents: list[Any],
        ephemeral: bool = False,
        max_team_size: int = 5,
        router_timeout: float = 30.0,
        max_iterations: int = 50,
    ) -> None:
        # Validate agents
        if not agents:
            raise ValueError("agents list cannot be empty")

        names = [getattr(g, "agentkit_name", None) for g in agents]
        if any(n is None for n in names):
            raise ValueError(
                "All agents must have agentkit_name (use the agent metaclass)"
            )

        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate agent names: {set(dupes)}")

        if max_team_size < 1:
            raise ValueError("max_team_size must be >= 1")

        self._agents_by_name: dict[str, Any] = {
            g.agentkit_name: g for g in agents
        }
        self._ephemeral = ephemeral
        self._max_team_size = max_team_size
        self._router_timeout = router_timeout
        self._max_iterations = max_iterations
        self._active_team: ActiveTeam | None = None
        self._compiled_cache: dict[str, Any] = {}

        # Build tools bound to this middleware instance
        from langchain_agentkit.tools.team import create_team_tools

        self._tools = create_team_tools(self)

    @property
    def tools(self) -> list[BaseTool]:
        """Team coordination tools."""
        return list(self._tools)

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> str:
        """Generate team coordination prompt section.

        If a team is active: shows roster with statuses and pending messages.
        If no team: shows available agents and guidelines.
        """
        # Build agent roster
        roster_lines = []
        for name, graph in self._agents_by_name.items():
            desc = getattr(graph, "agentkit_description", "")
            roster_lines.append(f"- **{name}**: {desc}" if desc else f"- **{name}**")
        agent_roster = "\n".join(roster_lines)

        # Load base template
        template_path = _PROMPTS_DIR / "team_coordination.md"
        if template_path.exists():
            template = PromptTemplate.from_file(template_path)
            base_prompt = template.format(agent_roster=agent_roster)
        else:
            base_prompt = f"## Team Coordination\n\n### Available Agents\n\n{agent_roster}"

        # If team is active, append live status
        if self._active_team is not None:
            team = self._active_team
            status_lines = [f"\n### Active Team: {team.name}\n"]

            for name, task in team.members.items():
                agent_type = team.member_types.get(name, "unknown")
                if task.done():
                    try:
                        task.result()
                        icon = "✅"
                    except asyncio.CancelledError:
                        icon = "🚫"
                    except Exception:
                        icon = "❌"
                else:
                    icon = "🔄"
                pending = team.bus.pending_count(name)
                pending_str = f" ({pending} pending)" if pending > 0 else ""
                status_lines.append(
                    f"- {icon} **{name}** ({agent_type}){pending_str}"
                )

            lead_pending = team.bus.pending_count("lead")
            if lead_pending > 0:
                status_lines.append(
                    f"\n⚠️ You have **{lead_pending} unread message(s)**. "
                    "Use CheckTeammates to retrieve them."
                )

            base_prompt += "\n".join(status_lines)

        return base_prompt

    @property
    def state_schema(self) -> type:
        """Team coordination requires ``TeamState`` in the graph state."""
        from langchain_agentkit.state import TeamState

        return TeamState
