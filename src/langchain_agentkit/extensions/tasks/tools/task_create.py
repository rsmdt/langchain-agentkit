"""TaskCreate tool."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from langchain_agentkit.extensions.tasks.tools.shared import (
    Task,
    _TaskCreateInput,
)

_TASK_CREATE_DESCRIPTION = """Create a structured task list for the current session to track multi-step work and show progress.

Use for non-trivial work (~3+ steps), when the user gives multiple tasks, or when asked for a todo list. Skip it for single, trivial, or conversational tasks — just do those.

Fields:
- subject — short imperative title ("Draft the outline").
- description — what to do, with context and acceptance criteria.
- active_form (optional) — present-continuous label shown while in_progress; defaults to subject.

Tasks start `pending`. Check TaskList first to avoid duplicates; use TaskUpdate to set dependencies."""


def _task_create_description(team_active: bool = False) -> str:
    """Return the TaskCreate description, optionally with team tips."""
    base = _TASK_CREATE_DESCRIPTION
    if team_active:
        base += (
            "\n\nTeam tips: to give this task to an existing teammate, create it, "
            "then set its owner via TaskUpdate. Creating a task is NOT creating a team "
            "— use TeamCreate only to spin up new agents."
        )
    return base


def _task_create(
    subject: str,
    description: str,
    state: dict[str, Any],
    tool_call_id: str,
    active_form: str = "",
) -> Command:  # type: ignore[type-arg]
    """Create a new task with pending status."""
    task: Task = {
        "id": str(uuid.uuid4()),
        "subject": subject,
        "description": description,
        "status": "pending",
        "active_form": active_form,
    }
    tasks = list(state.get("tasks") or [])
    tasks.append(task)
    return Command(
        update={
            "tasks": tasks,
            "messages": [ToolMessage(content=json.dumps(task), tool_call_id=tool_call_id)],
        }
    )


def build_task_create_tool(team_active: bool = False) -> Any:
    """Build the TaskCreate StructuredTool."""
    return StructuredTool.from_function(
        func=_task_create,
        name="TaskCreate",
        description=_task_create_description(team_active),
        args_schema=_TaskCreateInput,
        handle_tool_error=True,
    )
