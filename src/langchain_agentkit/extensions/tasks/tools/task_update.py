"""TaskUpdate tool."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.types import Command

from langchain_agentkit.extensions.tasks.tools.shared import (
    _apply_task_fields,
    _cascade_delete,
    _TaskUpdateInput,
    _validate_claim,
)

_TASK_UPDATE_DESCRIPTION = """Change a task's status or details. Use when a task's state or content changes, such as marking it done. Modifies one existing task."""


def _task_update(
    task_id: str,
    state: dict[str, Any],
    tool_call_id: str,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    owner: str | None = None,
    metadata: dict[str, Any] | None = None,
    add_blocked_by: list[str] | None = None,
    add_blocks: list[str] | None = None,
) -> Command:  # type: ignore[type-arg]
    """Update an existing task."""
    tasks = [dict(t) for t in (state.get("tasks") or [])]
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")

    _validate_claim(task, tasks, status, owner)

    if status == "deleted":
        _cascade_delete(task_id, tasks)

    _apply_task_fields(
        task,
        tasks,
        status=status,
        subject=subject,
        description=description,
        active_form=active_form,
        owner=owner,
        metadata=metadata,
        add_blocked_by=add_blocked_by,
        add_blocks=add_blocks,
    )

    return Command(
        update={
            "tasks": tasks,
            "messages": [ToolMessage(content=json.dumps(task), tool_call_id=tool_call_id)],
        }
    )


def build_task_update_tool() -> Any:
    """Build the TaskUpdate StructuredTool."""
    return StructuredTool.from_function(
        func=_task_update,
        name="TaskUpdate",
        description=_TASK_UPDATE_DESCRIPTION,
        args_schema=_TaskUpdateInput,
        handle_tool_error=True,
    )
