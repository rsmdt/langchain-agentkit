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


_SEND_MESSAGE_DESCRIPTION = """Send a message to a teammate.

Your plain text output is NOT visible to other agents — to communicate, you MUST call this tool. Messages from teammates are delivered to you automatically.

Use `to: "*"` to broadcast to all teammates — expensive, use only when everyone genuinely needs it. Refer to teammates by name.

Do NOT send structured JSON status messages. Communicate in plain text when you need to message teammates. Use TaskUpdate to mark tasks completed."""


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
