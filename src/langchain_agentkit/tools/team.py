"""Command-based team coordination tools for LangGraph agents.

Tools use ``InjectedState`` to read current team state and return
``Command(update={...})`` to apply changes. The ``AgentTeamMiddleware``
injects itself as a closure binding so tools can access the active team.

Usage::

    from langchain_agentkit.middleware.teams import AgentTeamMiddleware

    mw = AgentTeamMiddleware([researcher, coder])
    mw.tools  # [SpawnTeam, AssignTask, MessageTeammate, CheckTeammates, DissolveTeam]
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

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.middleware.teams import AgentTeamMiddleware


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _MemberSpec(BaseModel):
    """Specification for a single team member."""

    name: str = Field(description="Unique name for this team member.")
    agent_type: str = Field(description="Agent type — must match a registered agent.")


class _SpawnTeamInput(BaseModel):
    team_name: str = Field(description="Name for the team.")
    members: list[_MemberSpec] = Field(
        description=(
            "List of members to create. Each needs a unique name and "
            "an agent_type that matches a registered agent."
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


def _require_active_team(middleware: AgentTeamMiddleware) -> Any:
    """Return the active team or raise ToolException."""
    team = middleware._active_team  # noqa: SLF001
    if team is None:
        raise ToolException("No active team. Call SpawnTeam first.")
    return team


def _require_member(middleware: AgentTeamMiddleware, member_name: str) -> None:
    """Verify member exists in the active team."""
    team = _require_active_team(middleware)
    if member_name not in team.members:
        raise ToolException(f"Member '{member_name}' not in active team.")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _spawn_team(
    team_name: str,
    members: list[dict[str, Any]],
    state: dict[str, Any],
    tool_call_id: str,
    *,
    middleware: AgentTeamMiddleware,
) -> Command:  # type: ignore[type-arg]
    """Create a team with named members running as asyncio.Tasks."""
    from langchain_agentkit.middleware.teams import ActiveTeam, TeamMessageBus, _teammate_loop

    if middleware._active_team is not None:  # noqa: SLF001
        raise ToolException("Team already active. Dissolve first.")

    if not members:
        raise ToolException("Members list cannot be empty.")

    # Validate no duplicate member names
    names = [m["name"] if isinstance(m, dict) else m.name for m in members]
    if len(set(names)) != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        raise ToolException(f"Duplicate member names: {set(dupes)}")

    # Validate max team size
    if len(members) > middleware._max_team_size:  # noqa: SLF001
        raise ToolException(
            f"Team size {len(members)} exceeds maximum of {middleware._max_team_size}."  # noqa: SLF001
        )

    # Validate all agent_types exist
    agents_by_name = middleware._agents_by_name  # noqa: SLF001
    for member in members:
        agent_type = member["agent_type"] if isinstance(member, dict) else member.agent_type
        if agent_type not in agents_by_name:
            available = ", ".join(sorted(agents_by_name.keys()))
            raise ToolException(
                f"Agent type '{agent_type}' not found. Available: {available}"
            )

    # Create message bus and register all members + lead
    bus = TeamMessageBus()
    bus.register("lead")

    member_tasks: dict[str, asyncio.Task[str]] = {}
    member_types: dict[str, str] = {}
    team_members_state: list[dict[str, Any]] = []

    for member in members:
        member_name = member["name"] if isinstance(member, dict) else member.name
        agent_type = member["agent_type"] if isinstance(member, dict) else member.agent_type

        bus.register(member_name)

        # Compile graph with a per-member checkpointer for conversation history
        agent_graph = agents_by_name[agent_type]
        from langgraph.checkpoint.memory import InMemorySaver

        member_checkpointer = InMemorySaver()
        compiled = agent_graph.compile(checkpointer=member_checkpointer)
        thread_id = f"team-{team_name}-{member_name}"

        # Create asyncio.Task for the teammate loop
        task = asyncio.create_task(
            _teammate_loop(member_name, compiled, bus, thread_id=thread_id),
            name=thread_id,
        )
        member_tasks[member_name] = task
        member_types[member_name] = agent_type

        team_members_state.append({
            "name": member_name,
            "agent_type": agent_type,
            "status": "idle",
        })

    # Store active team on middleware
    middleware._active_team = ActiveTeam(  # noqa: SLF001
        name=team_name,
        bus=bus,
        members=member_tasks,
        member_types=member_types,
    )

    result = {
        "team_name": team_name,
        "members": [
            {
                "name": m["name"] if isinstance(m, dict) else m.name,
                "agent_type": m["agent_type"] if isinstance(m, dict) else m.agent_type,
            }
            for m in members
        ],
    }

    return Command(
        update={
            "team_members": team_members_state,
            "team_name": team_name,
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


async def _assign_task(
    member_name: str,
    task_description: str,
    state: dict[str, Any],
    tool_call_id: str,
    *,
    middleware: AgentTeamMiddleware,
) -> Command:  # type: ignore[type-arg]
    """Send a task to a team member via message bus."""
    _require_member(middleware, member_name)
    team = middleware._active_team  # noqa: SLF001

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
    middleware: AgentTeamMiddleware,
) -> Command:  # type: ignore[type-arg]
    """Send a message to a team member via the message bus."""
    _require_member(middleware, member_name)
    team = middleware._active_team  # noqa: SLF001

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
    middleware: AgentTeamMiddleware,
) -> Command:  # type: ignore[type-arg]
    """Check team member statuses and drain pending messages for lead."""
    team = _require_active_team(middleware)

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
            await team.bus.send("lead", member_name, "__shutdown__")

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
    middleware: AgentTeamMiddleware,
) -> Command:  # type: ignore[type-arg]
    """Gracefully shut down the team."""
    team = _require_active_team(middleware)

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
    middleware._active_team = None  # noqa: SLF001

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
- member names must be unique within the team
- agent_type must match a registered agent name
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


def create_team_tools(middleware: AgentTeamMiddleware) -> list[BaseTool]:
    """Create Command-based team coordination tools.

    Tools are bound to the middleware instance via closures so they can
    access the active team and message bus.

    Returns five tools: SpawnTeam, AssignTask, MessageTeammate,
    CheckTeammates, DissolveTeam.
    """

    return [
        StructuredTool.from_function(
            coroutine=partial(_spawn_team, middleware=middleware),
            name="SpawnTeam",
            description=_SPAWN_TEAM_DESCRIPTION,
            args_schema=_SpawnTeamInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_assign_task, middleware=middleware),
            name="AssignTask",
            description=_ASSIGN_TASK_DESCRIPTION,
            args_schema=_AssignTaskInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_message_teammate, middleware=middleware),
            name="MessageTeammate",
            description=_MESSAGE_TEAMMATE_DESCRIPTION,
            args_schema=_MessageTeammateInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_check_teammates, middleware=middleware),
            name="CheckTeammates",
            description=_CHECK_TEAMMATES_DESCRIPTION,
            args_schema=_CheckTeammatesInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_dissolve_team, middleware=middleware),
            name="DissolveTeam",
            description=_DISSOLVE_TEAM_DESCRIPTION,
            args_schema=_DissolveTeamInput,
            handle_tool_error=True,
        ),
    ]
