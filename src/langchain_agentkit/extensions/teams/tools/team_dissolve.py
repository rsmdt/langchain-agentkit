"""TeamDissolve tool."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from langchain_agentkit.extensions.teams.tools.shared import (
    _cleanup_bus,
    _require_active_team,
    _shutdown_team_tasks,
    _TeamDissolveInput,
    _unassign_teammate_tasks,
)

if TYPE_CHECKING:
    import asyncio

    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


_DISSOLVE_TEAM_DESCRIPTION = """Shut down the team and collect final results: signals all members, waits up to `timeout` seconds for graceful completion, then cancels the rest.

Use only when ALL work is done and you're ready to synthesize and respond — not for a mid-flight progress check (use TeamStatus for that)."""


async def _dissolve_team(
    state: dict[str, Any],
    tool_call_id: str,
    timeout: float = 30.0,
    *,
    ext: TeamExtension,
) -> Command:  # type: ignore[type-arg]
    """Gracefully shut down the team and clear cross-turn metadata."""
    from langchain_agentkit.extensions.teams.bus import task_status

    await ext.rehydrate_if_needed(state)

    async with ext.team_lock:
        team = _require_active_team(ext)

        await _shutdown_team_tasks(team, timeout)

        def _final_status(task: asyncio.Task[str]) -> str:
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
        ext._capture_buffer = []

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


def build_team_dissolve_tool(ext: TeamExtension) -> BaseTool:
    """Build the TeamDissolve StructuredTool."""
    return StructuredTool.from_function(
        coroutine=partial(_dissolve_team, ext=ext),
        name="TeamDissolve",
        description=_DISSOLVE_TEAM_DESCRIPTION,
        args_schema=_TeamDissolveInput,
        handle_tool_error=True,
    )
