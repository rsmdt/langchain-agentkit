"""Command-based team coordination tools for LangGraph agents.

Tools use ``InjectedState`` to read current team state and return
``Command(update={...})`` to apply changes. The ``TeamExtension``
injects itself as a closure binding so tools can access the active team.

Usage::

    from langchain_agentkit.extensions.teams import TeamExtension

    ext = TeamExtension(agents=[researcher, coder])
    ext.tools  # [TeamCreate, TeamMessage, TeamStatus, TeamDissolve]
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

from langchain_agentkit.extensions.agents.refs import Dynamic, Predefined

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


# ---------------------------------------------------------------------------
# Teammate system prompt addendum
# ---------------------------------------------------------------------------

_TEAMMATE_ADDENDUM = """\

# Agent Teammate Communication

IMPORTANT: You are running as an agent in a team. To communicate with anyone \
on your team:
- Use the TeamMessage tool with `to: "<name>"` to send messages to specific teammates
- Use the TeamMessage tool with `to: "*"` sparingly for team-wide broadcasts

Just writing a response in text is not visible to others on your team — \
you MUST use the TeamMessage tool.

# Task Management

You share a task list with your team. Use the standard task tools to coordinate:
- **TaskList** to see available work
- **TaskCreate** to add new tasks
- **TaskUpdate** to claim tasks (set owner), update status, or mark completed
- **TaskGet** to read full task details
"""


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


class _TeamCreateInput(BaseModel):
    name: str = Field(description="Name for the team.")
    agents: list[_AgentSpec] = Field(
        description=(
            "List of agents to include. Each needs a unique name and "
            "an agent reference ({id} or {prompt})."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamMessageInput(BaseModel):
    to: str = Field(
        description='Recipient: teammate name, or "*" for broadcast to all teammates.',
    )
    message: str = Field(description="Message content.")
    summary: str = Field(
        default="",
        description="5-10 word summary shown as preview (optional).",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamStatusInput(BaseModel):
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamDissolveInput(BaseModel):
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
        raise ToolException("No active team. Call TeamCreate first.")
    return team


def _require_member(ext: TeamExtension, member_name: str) -> None:
    """Verify member exists in the active team."""
    team = _require_active_team(ext)
    if member_name not in team.members:
        raise ToolException(f"Member '{member_name}' not in active team.")


# ---------------------------------------------------------------------------
# Predefined agent proxy compilation
# ---------------------------------------------------------------------------

_TASK_TOOL_NAMES = frozenset({"TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskStop"})


def _compile_with_proxy_tasks(
    agent_graph: Any,
    bus: Any,
    member_name: str,
) -> Any:
    """Compile a predefined agent graph, replacing task tools with proxies.

    If the graph was built by the ``Agent`` class, it carries
    ``_agentkit_*`` metadata that allows rebuilding with modified tools.
    If not, falls back to compiling as-is.

    No checkpointer is attached: the new teammate loop passes full
    history as input on each ``ainvoke`` call, so per-teammate
    ``InMemorySaver`` is unnecessary (and would require a
    ``thread_id`` config the loop no longer provides).
    """
    handler = getattr(agent_graph, "_agentkit_handler", None)
    llm = getattr(agent_graph, "_agentkit_llm", None)
    kit = getattr(agent_graph, "_agentkit_kit", None)
    original_tools = getattr(agent_graph, "_agentkit_user_tools", None)

    if handler is None or llm is None or kit is None:
        # Not an agentkit-built graph — compile as-is, no checkpointer
        return agent_graph.compile()

    from langchain_agentkit.extensions.teams.task_proxy import create_task_proxy_tools

    # Replace user-level task tools with proxies
    proxy_tools = create_task_proxy_tools(bus, member_name)
    filtered_user_tools = [t for t in (original_tools or []) if t.name not in _TASK_TOOL_NAMES]
    new_user_tools = filtered_user_tools + proxy_tools

    # Also strip task tools from kit extensions to avoid duplicates
    from langchain_agentkit.agent_kit import AgentKit
    from langchain_agentkit.extensions.tasks import TasksExtension

    filtered_extensions = [ext for ext in kit.extensions if not isinstance(ext, TasksExtension)]
    graph_name = getattr(agent_graph, "name", member_name)
    new_kit = AgentKit(
        extensions=filtered_extensions,
        prompt=kit.base_prompt,
        tools=new_user_tools,
        model=llm,
        name=graph_name,
    )

    rebuilt = new_kit.compile(handler)
    return rebuilt.compile()


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
    # Rehydrate first in case a previous-turn team was checkpointed.
    await ext.rehydrate_if_needed(state)
    # Lock guards against concurrent TeamCreate calls from parallel ToolNode execution
    async with ext.team_lock:
        return await _agent_team_inner(name, agents, state, tool_call_id, ext=ext)


def _parse_agent_ref(agent_spec: Any) -> tuple[str, str | None, str | None]:
    """Extract (member_name, agent_id, agent_prompt) from an LLM-supplied spec."""
    member_name = agent_spec["name"] if isinstance(agent_spec, dict) else agent_spec.name
    ref = agent_spec["agent"] if isinstance(agent_spec, dict) else agent_spec.agent
    if isinstance(ref, Dynamic):
        return member_name, None, ref.prompt
    if isinstance(ref, Predefined):
        return member_name, ref.id, None
    if isinstance(ref, dict):
        return member_name, ref.get("id"), ref.get("prompt")
    return member_name, getattr(ref, "id", None), getattr(ref, "prompt", None)


def _validate_team_creation(
    name: str,
    agents: list[dict[str, Any]],
    ext: TeamExtension,
) -> None:
    """Run pre-creation invariants; raise ``ToolException`` on violation."""
    from langchain_agentkit.extensions.agents.refs import resolve_agent_by_name

    if ext.active_team is not None:
        raise ToolException("Team already active. Dissolve first.")
    if not agents:
        raise ToolException("Agents list cannot be empty.")

    agent_names = [a["name"] if isinstance(a, dict) else a.name for a in agents]
    if len(set(agent_names)) != len(agent_names):
        dupes = [n for n in agent_names if agent_names.count(n) > 1]
        raise ToolException(f"Duplicate agent names: {set(dupes)}")
    if len(agents) > ext.max_team_size:
        raise ToolException(f"Team size {len(agents)} exceeds maximum of {ext.max_team_size}.")

    registered_agents = ext.agents_by_name
    for agent_spec in agents:
        _, agent_id, agent_prompt = _parse_agent_ref(agent_spec)
        if agent_prompt is not None:
            if not ext.ephemeral:
                raise ToolException(
                    "Dynamic/ephemeral agents are not enabled. "
                    "Set ephemeral=True on TeamExtension to allow custom agents."
                )
        elif agent_id is not None:
            resolve_agent_by_name(agent_id, registered_agents)


def _build_teammate_specs(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LLM-supplied agent specs into persistable ``TeammateSpec``s."""
    specs: list[dict[str, Any]] = []
    for agent_spec in agents:
        member_name, agent_id, agent_prompt = _parse_agent_ref(agent_spec)
        if agent_prompt is not None:
            specs.append(
                {
                    "member_name": member_name,
                    "kind": "dynamic",
                    "system_prompt": agent_prompt,
                },
            )
        else:
            specs.append(
                {
                    "member_name": member_name,
                    "kind": "predefined",
                    "agent_id": agent_id,
                },
            )
    return specs


async def _rollback_partial(
    member_tasks: dict[str, asyncio.Task[str]],
    bus: Any,
    ext: TeamExtension,
) -> None:
    """Cancel spawned tasks and clean up bus on partial team creation failure."""
    for partial_task in member_tasks.values():
        partial_task.cancel()
    if member_tasks:
        with contextlib.suppress(Exception):
            await asyncio.wait(list(member_tasks.values()), timeout=2.0)
    for member in list(member_tasks.keys()):
        with contextlib.suppress(Exception):
            bus.unregister(member)
    with contextlib.suppress(Exception):
        bus.unregister("lead")
    ext._capture_buffer = []


def _spawn_member(
    spec: dict[str, Any],
    bus: Any,
    team_name: str,
    ext: TeamExtension,
) -> tuple[asyncio.Task[str], str]:
    """Spawn a single teammate task. Returns (task, member_type)."""
    from langchain_agentkit.extensions.teams.bus import _teammate_loop

    member_name = spec["member_name"]
    bus.register(member_name)
    compiled = ext.build_teammate_graph(spec, bus)  # type: ignore[arg-type]
    task = asyncio.create_task(
        _teammate_loop(
            member_name,
            compiled,
            bus,
            initial_history=[],
            capture_buffer=ext._capture_buffer,
        ),
        name=f"team-{team_name}-{member_name}",
    )
    member_type = f"ephemeral:{member_name}" if spec["kind"] == "dynamic" else spec["agent_id"]
    return task, member_type


async def _agent_team_inner(
    name: str,
    agents: list[dict[str, Any]],
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Inner implementation of _agent_team, called under _team_lock."""
    from datetime import UTC, datetime

    from langchain_agentkit.extensions.teams.bus import (
        ActiveTeam,
        TeamMessageBus,
    )

    _validate_team_creation(name, agents, ext)
    specs = _build_teammate_specs(agents)

    bus = TeamMessageBus()
    bus.register("lead")
    ext._capture_buffer = []

    member_tasks: dict[str, asyncio.Task[str]] = {}
    member_types: dict[str, str] = {}

    try:
        for spec in specs:
            task, member_type = _spawn_member(spec, bus, name, ext)
            member_tasks[spec["member_name"]] = task
            member_types[spec["member_name"]] = member_type
    except Exception as exc:
        await _rollback_partial(member_tasks, bus, ext)
        raise ToolException(f"Failed to build team: {exc}") from exc

    ext.active_team = ActiveTeam(
        name=name,
        bus=bus,
        members=member_tasks,
        member_types=member_types,
    )

    result = {
        "team_name": name,
        "agents": [
            {
                "name": spec["member_name"],
                "agent": (
                    {"id": spec["agent_id"]}
                    if spec["kind"] == "predefined"
                    else {"prompt": spec["system_prompt"]}
                ),
            }
            for spec in specs
        ],
    }

    team_metadata: dict[str, Any] = {
        "name": name,
        "members": specs,
        "created_at": datetime.now(UTC).isoformat(),
    }

    return Command(
        update={
            "team": team_metadata,
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


async def _send_message(
    to: str,
    message: str,
    state: dict[str, Any],
    tool_call_id: str,
    summary: str = "",
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Send a message to a teammate or broadcast to all teammates."""
    await ext.rehydrate_if_needed(state)
    team = _require_active_team(ext)

    if to == "*":
        # Broadcast to all active team members (skip sender / lead)
        await team.bus.broadcast("lead", message)
        recipients = [n for n in team.members if n != "lead"]
        result = {"broadcast": True, "recipients": recipients, "message": message[:100]}
    else:
        _require_member(ext, to)
        await team.bus.send("lead", to, message)
        result = {"sent_to": to, "message": message[:100]}

    return Command(
        update={
            "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
        }
    )


def _collect_member_statuses(team: Any, agent_tasks: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Build per-member status dicts from asyncio.Task states."""
    statuses: list[dict[str, Any]] = []
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

        current_tasks = agent_tasks.get(name, [])
        statuses.append(
            {
                "name": name,
                "agent_type": team.member_types.get(name, "unknown"),
                "status": status,
                "work_status": "busy" if current_tasks else "idle",
                "current_tasks": current_tasks,
                "pending_messages": team.bus.pending_count(name),
            }
        )
    return statuses


async def _check_teammates(
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Check team member statuses and drain pending messages for lead."""
    await ext.rehydrate_if_needed(state)
    team = _require_active_team(ext)

    tasks = state.get("tasks") or []

    # Derive task-based work status per agent
    agent_tasks: dict[str, list[str]] = {}
    for t in tasks:
        owner = t.get("owner")
        if owner and t.get("status") not in ("completed", "deleted"):
            agent_tasks.setdefault(owner, []).append(t["id"])

    # Gather member statuses from asyncio.Task states
    member_statuses = _collect_member_statuses(team, agent_tasks)

    # Drain pending messages for lead
    lead_messages: list[dict[str, Any]] = []
    while True:
        msg = await team.bus.receive("lead", timeout=0.1)
        if msg is None:
            break
        lead_messages.append(
            {
                "from": msg.sender,
                "content": msg.content,
            }
        )

    # Include task progress from state
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

    return Command(
        update={
            "messages": [ToolMessage(content=json.dumps(report), tool_call_id=tool_call_id)],
        }
    )


async def _shutdown_team_tasks(
    team: Any,
    timeout: float,
) -> None:
    """Send structured shutdown requests and wait for tasks to finish."""
    shutdown_msg = json.dumps({"type": "shutdown_request"})
    for member_name in team.members:
        with contextlib.suppress(Exception):
            await team.bus.send("lead", member_name, shutdown_msg)

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


def _unassign_teammate_tasks(
    tasks: list[dict[str, Any]],
    member_names: list[str],
) -> list[dict[str, Any]]:
    """Reset unresolved tasks owned by dissolved teammates to pending.

    Returns a new list with ownership cleared on affected tasks.
    """
    name_set = set(member_names)
    result = []
    for t in tasks:
        t = dict(t)
        owner = t.get("owner")
        if owner and owner in name_set and t.get("status") not in ("completed", "deleted"):
            t["status"] = "pending"
            t["owner"] = None
        result.append(t)
    return result


async def _dissolve_team(
    state: dict[str, Any],
    tool_call_id: str,
    timeout: float = 30.0,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Gracefully shut down the team and clear cross-turn metadata."""
    from langchain_agentkit.extensions.teams.bus import task_status

    # Rehydrate any checkpointed team so we can dissolve it cleanly.
    await ext.rehydrate_if_needed(state)

    # Lock guards against concurrent dissolve/create from parallel ToolNode execution
    async with ext.team_lock:
        team = _require_active_team(ext)

        await _shutdown_team_tasks(team, timeout)

        def _final_status(task: asyncio.Task[str]) -> str:
            # After shutdown + cancel, any still-running task is treated as cancelled.
            status = task_status(task)
            return "cancelled" if status == "running" else status

        final_members: list[dict[str, Any]] = [
            {
                "name": name,
                "agent_type": team.member_types.get(name, "unknown"),
                "status": _final_status(task),
            }
            for name, task in team.members.items()
        ]

        team_name = team.name
        _cleanup_bus(team)
        ext.active_team = None
        # Clear the per-turn capture buffer — anything produced before
        # dissolve has already been persisted via before_model flushes.
        ext._capture_buffer = []

    # Unassign tasks owned by dissolved teammates
    member_names = [m["name"] for m in final_members]
    tasks = list(state.get("tasks") or [])
    updated_tasks = _unassign_teammate_tasks(tasks, member_names)

    result = {
        "dissolved": True,
        "team_name": team_name,
        "final_statuses": final_members,
    }

    update: dict[str, Any] = {
        "team": None,
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)],
    }
    if updated_tasks:
        update["tasks"] = updated_tasks

    return Command(update=update)


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

Each member runs as an independent agent. You coordinate by sending messages \
and checking status. Members report results back automatically.

Important:
- agent names must be unique within the team
- Use {id: name} for predefined agents, {prompt: text} for ephemeral
- Only one team can be active at a time\
"""

_SEND_MESSAGE_DESCRIPTION = """\
Send a message to a teammate.

Your plain text output is NOT visible to other agents — to communicate, \
you MUST call this tool. Messages from teammates are delivered automatically.

Use `to: "*"` to broadcast to all teammates — expensive, use only when \
everyone genuinely needs it. Refer to teammates by name.

Do NOT send structured JSON status messages. Just communicate in plain text \
when you need to message teammates. Use TaskUpdate to mark tasks completed.\
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

    Returns four tools: TeamCreate, TeamMessage, TeamStatus, TeamDissolve.
    """

    return [
        StructuredTool.from_function(
            coroutine=partial(_agent_team, ext=ext),
            name="TeamCreate",
            description=_SPAWN_TEAM_DESCRIPTION,
            args_schema=_TeamCreateInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_send_message, ext=ext),
            name="TeamMessage",
            description=_SEND_MESSAGE_DESCRIPTION,
            args_schema=_TeamMessageInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_check_teammates, ext=ext),
            name="TeamStatus",
            description=_CHECK_TEAMMATES_DESCRIPTION,
            args_schema=_TeamStatusInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_dissolve_team, ext=ext),
            name="TeamDissolve",
            description=_DISSOLVE_TEAM_DESCRIPTION,
            args_schema=_TeamDissolveInput,
            handle_tool_error=True,
        ),
    ]
