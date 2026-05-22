"""TeamMessage tool."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from langchain_agentkit.extensions.teams.tools.shared import (
    _require_active_team,
    _require_member,
    _TeamMessageInput,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


_SEND_MESSAGE_DESCRIPTION = """Tell a teammate what to do — assign work, give an instruction, or hand off a task to them by name. This is the ONLY way to direct a teammate: your plain-text output is not visible to other agents, and their replies reach you automatically.

Use `to: "*"` to broadcast (expensive — only when everyone needs it). Send plain text, not JSON status; use TaskUpdate to mark tasks done."""


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


def build_team_message_tool(ext: TeamExtension) -> BaseTool:
    """Build the TeamMessage StructuredTool."""
    return StructuredTool.from_function(
        coroutine=partial(_send_message, ext=ext),
        name="TeamMessage",
        description=_SEND_MESSAGE_DESCRIPTION,
        args_schema=_TeamMessageInput,
        handle_tool_error=True,
    )
