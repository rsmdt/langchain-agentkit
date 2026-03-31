"""Command-based team coordination tools for LangGraph agents.

Tools use ``InjectedState`` to read current team state and return
``Command(update={...})`` to apply changes. The ``TeamExtension``
injects itself as a closure binding so tools can access the active team.

Usage::

    from langchain_agentkit.extensions.teams import TeamExtension

    mw = TeamExtension([researcher, coder])
    mw.tools  # [AgentTeam, AssignTask, MessageTeammate, CheckTeammates, DissolveTeam]
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_agentkit.tools.agent import Dynamic, Predefined

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams import TeamExtension


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _AgentSpec(BaseModel):
    """Specification for a single team agent."""

    name: str = Field(description="Unique name for this agent in the team.")
    agent: Predefined | Dynamic = Field(
        description=(
            "Agent to use. Use {id: name} to select a pre-defined agent "
            "from the roster, or {prompt: text} to create an ephemeral "
            "reasoning agent for this team role."
        ),
    )


class _AgentTeamInput(BaseModel):
    name: str = Field(description="Name for the team.")
    agents: list[_AgentSpec] = Field(
        description=(
            "List of agents to include. Each needs a unique name and "
            "an agent reference ({id} or {prompt})."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _AssignTaskInput(BaseModel):
    member_name: str = Field(description="Team member to assign work to.")
    task_description: str = Field(description="Clear, specific task to assign.")
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _MessageTeammateInput(BaseModel):
    member_name: str = Field(description="Team member to message.")
    message: str = Field(description="Message content.")
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _CheckTeammatesInput(BaseModel):
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _DissolveTeamInput(BaseModel):
    timeout: float = Field(
        default=30.0,
        description="Max seconds to wait for graceful shutdown.",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_active_team(ext: TeamExtension) -> Any:
    """Return the active team or raise ToolException."""
    team = ext.active_team
    if team is None:
        raise ToolException("No active team. Call AgentTeam first.")
    return team


def _require_member(ext: TeamExtension, member_name: str) -> None:
    """Verify member exists in the active team."""
    team = _require_active_team(ext)
    if member_name not in team.members:
        raise ToolException(f"Member '{member_name}' not in active team.")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _agent_team(
    name: str,
    agents: list[dict[str, Any]],
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Create a team with named agents running as asyncio.Tasks."""
    # Lock guards against concurrent AgentTeam calls from parallel ToolNode execution
    async with ext.team_lock:
        return await _agent_team_inner(name, agents, state, tool_call_id, ext=ext)


async def _agent_team_inner(
    name: str,
    agents: list[dict[str, Any]],
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Inner implementation of _agent_team, called under _team_lock."""
    from langchain_agentkit.extensions.teams import ActiveTeam, TeamMessageBus, _teammate_loop

    if ext.active_team is not None:
        raise ToolException("Team already active. Dissolve first.")

    if not agents:
        raise ToolException("Agents list cannot be empty.")

    # Validate no duplicate agent names
    agent_names = [a["name"] if isinstance(a, dict) else a.name for a in agents]
    if len(set(agent_names)) != len(agent_names):
        dupes = [n for n in agent_names if agent_names.count(n) > 1]
        raise ToolException(f"Duplicate agent names: {set(dupes)}")

    # Validate max team size
    if len(agents) > ext.max_team_size:
        raise ToolException(
            f"Team size {len(agents)} exceeds maximum of {ext.max_team_size}."
        )

    # Resolve agent references
    registered_agents = ext.agents_by_name
    from langchain_agentkit.extensions import resolve_agent

    def _parse_ref(agent_spec: Any) -> tuple[str, str | None, str | None]:
        """Extract (member_name, agent_id, agent_prompt) from a spec."""
        member_name = agent_spec["name"] if isinstance(agent_spec, dict) else agent_spec.name
        ref = agent_spec["agent"] if isinstance(agent_spec, dict) else agent_spec.agent
        if isinstance(ref, Dynamic):
            return member_name, None, ref.prompt
        if isinstance(ref, Predefined):
            return member_name, ref.id, None
        # Dict from LLM tool call
        if isinstance(ref, dict):
            return member_name, ref.get("id"), ref.get("prompt")
        return member_name, getattr(ref, "id", None), getattr(ref, "prompt", None)

    for agent_spec in agents:
        _, agent_id, agent_prompt = _parse_ref(agent_spec)

        if agent_prompt is not None:
            ephemeral_enabled = ext.ephemeral
            if not ephemeral_enabled:
                raise ToolException(
                    "Dynamic/ephemeral agents are not enabled. "
                    "Set ephemeral=True on TeamExtension to allow custom agents."
                )
        elif agent_id is not None:
            resolve_agent(agent_id, registered_agents)

    # Create message bus and register all agents + lead
    bus = TeamMessageBus()
    bus.register("lead")

    member_tasks: dict[str, asyncio.Task[str]] = {}
    member_types: dict[str, str] = {}
    team_agents_state: list[dict[str, Any]] = []

    for agent_spec in agents:
        member_name, agent_id, agent_prompt = _parse_ref(agent_spec)

        bus.register(member_name)

        if agent_prompt is not None:
            # Ephemeral agent — build on-the-fly graph
            from langchain_agentkit._graph_builder import build_graph
            from langchain_agentkit.agent_kit import AgentKit

            parent_llm_getter = ext.parent_llm_getter
            if parent_llm_getter is not None:
                ephemeral_llm = parent_llm_getter()
            else:
                raise ToolException("No parent LLM available for ephemeral agent.")

            async def _ephemeral_handler(
                handler_state: dict[str, Any],
                *,
                llm: Any,
                prompt: str,
                _sender: str = member_name,
                **kwargs: Any,
            ) -> dict[str, Any]:
                from langchain_core.messages import SystemMessage

                messages = [SystemMessage(content=prompt)] + handler_state["messages"]
                response = await llm.ainvoke(messages)
                return {"messages": [response], "sender": _sender}

            ephemeral_graph = build_graph(
                name=member_name,
                handler=_ephemeral_handler,
                llm=ephemeral_llm,
                user_tools=[],
                kit=AgentKit(extensions=[], prompt=agent_prompt),
            )
            from langgraph.checkpoint.memory import InMemorySaver

            compiled = ephemeral_graph.compile(checkpointer=InMemorySaver())
            agent_type_label = f"ephemeral:{member_name}"
        else:
            # Predefined agent — resolve and compile
            agent_target = registered_agents[agent_id]
            from langchain_agentkit.composability import AgentLike

            if isinstance(agent_target, AgentLike):
                compiled = agent_target
            else:
                from langgraph.checkpoint.memory import InMemorySaver

                compiled = agent_target.compile(checkpointer=InMemorySaver())
            agent_type_label = agent_id

        thread_id = f"team-{name}-{member_name}"

        task = asyncio.create_task(
            _teammate_loop(member_name, compiled, bus, thread_id=thread_id),
            name=thread_id,
        )
        member_tasks[member_name] = task
        member_types[member_name] = agent_type_label

        team_agents_state.append({
            "name": member_name,
            "agent": {"id": agent_id} if agent_id else {"prompt": agent_prompt},
            "status": "idle",
        })

    # Store active team on extension
    ext.active_team = ActiveTeam(
        name=name,
        bus=bus,
        members=member_tasks,
        member_types=member_types,
    )

    result = {
        "team_name": name,
        "agents": [
            {"name": a["name"], "agent": a["agent"]}
            for a in team_agents_state
        ],
    }

    return Command(
        update={
            "team_members": [
                {"name": a["name"], "agent_type": member_types.get(a["name"], ""), "status": "idle"}
                for a in team_agents_state
            ],
            "team_name": name,
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


async def _assign_task(
    member_name: str,
    task_description: str,
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Send a task to a team member via message bus."""
    _require_member(ext, member_name)
    team = ext.active_team

    # Send task to member via bus
    await team.bus.send("lead", member_name, task_description)

    result = {"sent_to": member_name, "task": task_description[:80]}
    return Command(
        update={
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


async def _message_teammate(
    member_name: str,
    message: str,
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Send a message to a team member via the message bus."""
    _require_member(ext, member_name)
    team = ext.active_team

    await team.bus.send("lead", member_name, message)

    result = {"sent_to": member_name, "message": message[:100]}
    return Command(
        update={
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


async def _check_teammates(
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Check team member statuses and drain pending messages for lead."""
    team = _require_active_team(ext)

    # Gather member statuses from asyncio.Task states
    member_statuses: list[dict[str, Any]] = []
    for name, task in team.members.items():
        if task.done():
            try:
                task.result()
                status = "completed"
            except asyncio.CancelledError:
                status = "cancelled"
            except Exception as exc:
                status = f"failed: {exc}"
        else:
            status = "running"

        member_statuses.append({
            "name": name,
            "agent_type": team.member_types.get(name, "unknown"),
            "status": status,
            "pending_messages": team.bus.pending_count(name),
        })

    # Drain pending messages for lead
    lead_messages: list[dict[str, Any]] = []
    while True:
        msg = await team.bus.receive("lead", timeout=0.1)
        if msg is None:
            break
        lead_messages.append({
            "from": msg.sender,
            "content": msg.content,
        })

    # Include task progress from state
    tasks = state.get("tasks") or []
    task_summary = [
        {
            "id": t["id"],
            "subject": t.get("subject", ""),
            "status": t.get("status", "pending"),
            "owner": t.get("owner", ""),
        }
        for t in tasks
        if t.get("status") != "deleted"
    ]

    report = {
        "team_name": team.name,
        "members": member_statuses,
        "pending_messages": lead_messages,
        "tasks": task_summary,
    }

    # Build state update: inject lead messages as HumanMessages
    update: dict[str, Any] = {
        "messages": [ToolMessage(content=json.dumps(report), tool_call_id=tool_call_id)],
    }

    # Update team_members state with latest statuses
    team_members_state = [
        {"name": ms["name"], "agent_type": ms["agent_type"], "status": ms["status"]}
        for ms in member_statuses
    ]
    update["team_members"] = team_members_state

    return Command(update=update)


def _task_final_status(task: asyncio.Task[str]) -> str:
    """Determine the final status string for a completed asyncio.Task."""
    if not task.done():
        return "cancelled"
    try:
        task.result()
        return "completed"
    except asyncio.CancelledError:
        return "cancelled"
    except Exception:
        return "failed"


async def _shutdown_team_tasks(
    team: Any,
    timeout: float,
) -> None:
    """Send shutdown signals and wait for tasks to finish."""
    for member_name in team.members:
        with contextlib.suppress(Exception):
            from langchain_agentkit.extensions.teams import SHUTDOWN_SIGNAL

            await team.bus.send("lead", member_name, SHUTDOWN_SIGNAL)

    pending_tasks = [t for t in team.members.values() if not t.done()]
    if pending_tasks:
        _done, still_pending = await asyncio.wait(pending_tasks, timeout=timeout)
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.wait(still_pending, timeout=5.0)


def _cleanup_bus(team: Any) -> None:
    """Unregister all agents from the message bus."""
    for name in list(team.members.keys()):
        with contextlib.suppress(Exception):
            team.bus.unregister(name)
    with contextlib.suppress(Exception):
        team.bus.unregister("lead")


async def _dissolve_team(
    state: dict[str, Any],
    tool_call_id: str,
    timeout: float = 30.0,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Gracefully shut down the team."""
    # Lock guards against concurrent dissolve/create from parallel ToolNode execution
    async with ext.team_lock:
        team = _require_active_team(ext)

        await _shutdown_team_tasks(team, timeout)

        final_members: list[dict[str, Any]] = [
            {
                "name": name,
                "agent_type": team.member_types.get(name, "unknown"),
                "status": _task_final_status(task),
            }
            for name, task in team.members.items()
        ]

        _cleanup_bus(team)
        ext.active_team = None

    result = {
        "dissolved": True,
        "team_name": team.name,
        "final_statuses": final_members,
    }

    return Command(
        update={
            "team_members": final_members,
            "team_name": None,
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


# ---------------------------------------------------------------------------
# Tool descriptions
# ---------------------------------------------------------------------------


_SPAWN_TEAM_DESCRIPTION = """\
Create a team of concurrent agents for complex, multi-step work.

Use when:
- Work requires back-and-forth coordination between specialists
- Tasks have dependencies — one member's output informs another's work
- You need to steer work in progress based on intermediate results
- The project is too complex for a single delegation

Each member runs as an independent agent. You coordinate by assigning tasks \
and sending messages. Members report results back automatically.

Important:
- agent names must be unique within the team
- Use {id: name} for predefined agents, {prompt: text} for ephemeral
- Only one team can be active at a time\
"""

_ASSIGN_TASK_DESCRIPTION = """\
Send a task to a specific team member via the message bus.

The member receives the message and begins working. To track this task,
create it first with TaskCreate (set owner to the member name), then
use AssignTask to send the work description.

Tips:
- Write clear, specific descriptions with all needed context
- Use TaskCreate before AssignTask if you want task tracking
- Use CheckTeammates to monitor progress after assigning\
"""

_MESSAGE_TEAMMATE_DESCRIPTION = """\
Send a message to a team member.

Use to:
- Provide guidance or clarification to a working member
- Forward information from one member to another
- Unblock a member who is waiting for input
- Send follow-up instructions after reviewing their work

The member receives the message and can respond through the bus.\
"""

_CHECK_TEAMMATES_DESCRIPTION = """\
Check status of all team members and collect pending messages.

Returns:
- Each member's current status (running, completed, failed, cancelled)
- Any pending messages from members to you (the lead)
- Current task progress

Use frequently to:
- Monitor team progress
- Collect results from completed work
- Identify members that need guidance or are stuck\
"""

_DISSOLVE_TEAM_DESCRIPTION = """\
Shut down the team and collect final results.

Sends shutdown signal to all members, waits up to timeout seconds for \
graceful completion, then cancels remaining work.

Use when:
- All assigned work is complete
- You need to synthesize results and respond to the user
- A member has failed and recovery is not possible

After dissolving, synthesize all results before responding to the user.\
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_team_tools(ext: TeamExtension) -> list[BaseTool]:
    """Create Command-based team coordination tools.

    Tools are bound to the extension instance via closures so they can
    access the active team and message bus.

    Returns five tools: AgentTeam, AssignTask, MessageTeammate,
    CheckTeammates, DissolveTeam.
    """

    return [
        StructuredTool.from_function(
            coroutine=partial(_agent_team, ext=ext),
            name="AgentTeam",
            description=_SPAWN_TEAM_DESCRIPTION,
            args_schema=_AgentTeamInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_assign_task, ext=ext),
            name="AssignTask",
            description=_ASSIGN_TASK_DESCRIPTION,
            args_schema=_AssignTaskInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_message_teammate, ext=ext),
            name="MessageTeammate",
            description=_MESSAGE_TEAMMATE_DESCRIPTION,
            args_schema=_MessageTeammateInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_check_teammates, ext=ext),
            name="CheckTeammates",
            description=_CHECK_TEAMMATES_DESCRIPTION,
            args_schema=_CheckTeammatesInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_dissolve_team, ext=ext),
            name="DissolveTeam",
            description=_DISSOLVE_TEAM_DESCRIPTION,
            args_schema=_DissolveTeamInput,
            handle_tool_error=True,
        ),
    ]
