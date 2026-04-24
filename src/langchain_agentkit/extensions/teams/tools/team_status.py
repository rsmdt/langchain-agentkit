"""TeamStatus tool."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from langchain_agentkit.extensions.teams.tools.shared import (
    _require_active_team,
    _TeamStatusInput,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


_CHECK_TEAMMATES_DESCRIPTION = """Check status of all team members and collect pending messages.

Returns:
- Each member's current status (running, completed, failed, cancelled)
- Any pending messages from members to you (the lead)
- Current task progress

Use frequently to:
- Monitor team progress
- Collect results from completed work
- Identify members that need guidance or are stuck"""


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

    agent_tasks: dict[str, list[str]] = {}
    for t in tasks:
        owner = t.get("owner")
        if owner and t.get("status") not in ("completed", "deleted"):
            agent_tasks.setdefault(owner, []).append(t["id"])

    member_statuses = _collect_member_statuses(team, agent_tasks)

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


def build_team_status_tool(ext: TeamExtension) -> BaseTool:
    """Build the TeamStatus StructuredTool."""
    return StructuredTool.from_function(
        coroutine=partial(_check_teammates, ext=ext),
        name="TeamStatus",
        description=_CHECK_TEAMMATES_DESCRIPTION,
        args_schema=_TeamStatusInput,
        handle_tool_error=True,
    )
