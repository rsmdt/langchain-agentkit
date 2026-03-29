"""TeamExtension — message-driven team coordination.

Teammates run as asyncio.Tasks. The lead coordinates via tools
(SpawnTeam, AssignTask, MessageTeammate, CheckTeammates, DissolveTeam).
Communication flows through a ``TeamMessageBus`` backed by asyncio.Queue.

Usage::

    from langchain_agentkit import agent, TeamExtension

    class project_lead(agent):
        llm = ChatOpenAI(model="gpt-4o")
        extensions = [TeamExtension([researcher, coder])]
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

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_team_coordination_template = PromptTemplate.from_file(_PROMPTS_DIR / "team_coordination.md")


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
    thread_id: str | None = None,
) -> str:
    """Event loop for a single teammate.

    Blocks on ``receive()``, processes messages via the compiled graph,
    and sends results back to the sender. Exits on ``"__shutdown__"`` signal.

    If the compiled graph has a checkpointer and a ``thread_id`` is provided,
    conversation history accumulates automatically across messages via
    LangGraph's state persistence. Each ``ainvoke`` resumes from the
    previous state on the same thread.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    while True:
        msg = await message_bus.receive(member_name, timeout=30.0)
        if msg is None:
            continue  # idle, waiting for work
        if msg.content == "__shutdown__":
            return "shutdown"

        # Execute the full ReAct loop — checkpointer accumulates history
        try:
            result = await compiled_graph.ainvoke(
                {
                    "messages": [HumanMessage(content=msg.content)],
                    "sender": member_name,
                },
                config=config,
            )
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
    """Runtime state for an active team — lives on the extensions instance."""

    name: str
    bus: TeamMessageBus
    members: dict[str, asyncio.Task[str]]  # name → asyncio.Task
    member_types: dict[str, str]  # name → agent_type


# ---------------------------------------------------------------------------
# TeamExtension
# ---------------------------------------------------------------------------


class TeamExtension(Extension):
    """Extension providing message-driven team coordination.

    Teammates run as asyncio.Tasks. The lead reacts to teammate messages
    (not polls). Modeled after Claude Code's Agent Team pattern with
    AutoGen-inspired message dispatch primitives.

    Args:
        agents: List of StateGraph objects with ``.agentkit_name`` and
            ``.agentkit_description`` attributes (from the agent metaclass).
        max_team_size: Maximum number of team members allowed.
        router_timeout: Seconds to wait for messages in the Router Node.

    Note:
        ``AssignTask`` writes to the shared ``tasks`` state. Add
        ``TasksExtension`` to your extensions list if you want the lead
        to also have task management tools (TaskCreate, TaskList, etc.).
        There is one shared task list — no separate task system for teams.

    Example::

        from langchain_agentkit import agent, TeamExtension, TasksExtension

        class lead(agent):
            llm = ChatOpenAI(model="gpt-4o")
            extensions = [TasksExtension(), TeamExtension([researcher, coder])]
            ...
    """

    def __init__(
        self,
        agents: list[Any],
        max_team_size: int = 5,
        router_timeout: float = 30.0,
    ) -> None:
        from langchain_agentkit.extensions import validate_agent_list

        if max_team_size < 1:
            raise ValueError("max_team_size must be >= 1")

        self._agents_by_name: dict[str, Any] = validate_agent_list(agents)
        self._max_team_size = max_team_size
        self._router_timeout = router_timeout
        self._active_team: ActiveTeam | None = None

        # Build tools bound to this extensions instance
        from langchain_agentkit.tools.team import create_team_tools

        self._tools = tuple(create_team_tools(self))

    @property
    def tools(self) -> list[BaseTool]:
        """Team coordination tools."""
        return self._tools

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

        # Use cached template
        base_prompt = _team_coordination_template.format(agent_roster=agent_roster)

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

    def dependencies(self) -> list:
        """Team coordination requires TasksExtension for task tracking."""
        from langchain_agentkit.extensions.tasks import TasksExtension

        return [TasksExtension()]

    @property
    def state_schema(self) -> type:
        """Team coordination requires TeamState in the graph state."""
        from langchain_agentkit.state import TeamState

        return TeamState

    def graph_modifier(self, workflow: Any, node_name: str) -> Any:
        """Inject the Router Node into the graph topology.

        Adds a "router" node and its conditional edges. The ``_build_graph``
        function detects the router node and wires ``tools → router``
        instead of ``tools → handler``.

        Before (standard ReAct):
            handler → tools → handler → ... → END

        After (team-aware):
            handler → tools → router → handler (if messages) → ...
                                      → END    (if team dissolved or idle)
        """
        from langgraph.graph import END

        mw = self  # capture for closure

        async def _router_node(state: dict[str, Any]) -> dict[str, Any]:
            """Check message bus and inject teammate messages into state."""
            team = mw._active_team
            if team is None:
                # No team active — pass through (let handler synthesize)
                return {}  # _router_should_continue handles routing

            # Drain all pending messages for the lead
            messages: list[TeamMessage] = []
            while True:
                msg = await team.bus.receive("lead", timeout=0.1)
                if msg is None:
                    break
                messages.append(msg)

            if messages:
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[Message from teammate '{m.sender}']: {m.content}",
                            additional_kwargs={
                                "sender": m.sender,
                                "type": "teammate_message",
                            },
                        )
                        for m in messages
                    ]
                }

            # No messages yet — check if teammates are still working
            active_count = sum(
                1 for t in team.members.values() if not t.done()
            )
            if active_count == 0:
                return {}  # all done, _router_should_continue routes to END

            # Teammates still working — wait for a message
            msg = await team.bus.receive("lead", timeout=mw._router_timeout)
            if msg is not None:
                return {
                    "messages": [
                        HumanMessage(
                            content=f"[Message from teammate '{msg.sender}']: {msg.content}",
                            additional_kwargs={
                                "sender": msg.sender,
                                "type": "teammate_message",
                            },
                        )
                    ]
                }

            return {}

        def _router_should_continue(state: dict[str, Any]) -> str:
            """Route after Router Node: back to handler if messages, END if done."""
            team = mw._active_team

            # Team just dissolved — route back to handler for final synthesis
            if team is None:
                msgs = state.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    # If last message is a ToolMessage (from DissolveTeam), let
                    # the handler produce a final human-facing response
                    if hasattr(last, "type") and last.type == "tool":
                        return node_name
                return END

            # Check if new messages were just injected
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                kwargs = getattr(last, "additional_kwargs", {})
                if kwargs.get("type") == "teammate_message":
                    return node_name

            # All teammates done and no pending messages → END
            active_count = sum(1 for t in team.members.values() if not t.done())
            lead_pending = team.bus.pending_count("lead")
            if active_count == 0 and lead_pending == 0:
                return END

            return node_name

        # Add router node — _build_graph detects this and wires tools → router
        workflow.add_node("router", _router_node)
        workflow.add_conditional_edges(
            "router",
            _router_should_continue,
            {node_name: node_name, END: END},
        )

        return workflow
