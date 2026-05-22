"""TaskStop tool."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.types import Command

from langchain_agentkit.extensions.tasks.tools.shared import _TaskStopInput

_TASK_STOP_DESCRIPTION = """Stop an `in_progress` task by ID, resetting it to `pending` so another owner (or you, later) can pick it up.

Use to release a task you claimed but can't finish now, or to return a silent owner's task to the queue."""


def _task_stop(
    task_id: str,
    state: dict[str, Any],
    tool_call_id: str,
) -> Command:  # type: ignore[type-arg]
    """Stop a running task by setting its status back to pending."""
    tasks = [dict(t) for t in (state.get("tasks") or [])]
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")
    if task.get("status") != "in_progress":
        raise ToolException(
            f"Task '{task_id}' is not in_progress (status: {task.get('status')}).",
        )
    task["status"] = "pending"
    return Command(
        update={
            "tasks": tasks,
            "messages": [
                ToolMessage(
                    content=json.dumps({"id": task_id, "status": "pending", "stopped": True}),
                    tool_call_id=tool_call_id,
                ),
            ],
        }
    )


def build_task_stop_tool() -> Any:
    """Build the TaskStop StructuredTool."""
    return StructuredTool.from_function(
        func=_task_stop,
        name="TaskStop",
        description=_TASK_STOP_DESCRIPTION,
        args_schema=_TaskStopInput,
        handle_tool_error=True,
    )
