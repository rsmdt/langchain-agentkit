"""TeamCreate tool."""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.types import Command

from langchain_agentkit.extensions.agents.refs import Dynamic, Predefined
from langchain_agentkit.extensions.teams.tools._shared import _TeamCreateInput

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


_SPAWN_TEAM_DESCRIPTION = """Create a team of concurrent agents for complex, multi-step work.

Use when:
- Work requires back-and-forth coordination between specialists
- Tasks have dependencies — one member's output informs another's work
- You need to steer work in progress based on intermediate results
- The project is too complex for a single delegation

Each member runs as an independent agent. You coordinate by sending messages and checking status. Members return their results back automatically.

Important:
- Agent names must be unique within the team
- Use {id: name} for a pre-defined agent from the roster, {prompt: text} for an ephemeral reasoning agent
- Only one team can be active at a time"""


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


async def _agent_team(
    name: str,
    agents: list[dict[str, Any]],
    state: dict[str, Any],
    tool_call_id: str,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Create a team with named agents running as asyncio.Tasks."""
    await ext.rehydrate_if_needed(state)
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


def build_team_create_tool(ext: TeamExtension) -> BaseTool:
    """Build the TeamCreate StructuredTool."""
    return StructuredTool.from_function(
        coroutine=partial(_agent_team, ext=ext),
        name="TeamCreate",
        description=_SPAWN_TEAM_DESCRIPTION,
        args_schema=_TeamCreateInput,
        handle_tool_error=True,
    )
