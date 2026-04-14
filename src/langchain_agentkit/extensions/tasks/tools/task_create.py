"""TaskCreate tool."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from langchain_agentkit.extensions.tasks.tools._shared import (
    Task,
    _TaskCreateInput,
)

_TASK_CREATE_DESCRIPTION = """Use this tool to create a structured task list for your current session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool

Use this tool proactively in these scenarios:

- Complex multi-step tasks — When a task requires 3 or more distinct steps or actions
- Non-trivial and complex tasks — Tasks that require careful planning or multiple operations
- User explicitly requests a todo list — When the user directly asks you to use a task list
- User provides multiple tasks — When users provide a list of things to be done (numbered or comma-separated)
- After receiving new instructions — Immediately capture user requirements as tasks
- When you start working on a task — Mark it as in_progress BEFORE beginning work
- After completing a task — Mark it as completed and add any new follow-up tasks discovered along the way

## When NOT to Use This Tool

Skip using this tool when:
- There is only a single, straightforward task
- The task is trivial and tracking it provides no organizational benefit
- The task can be completed in less than 3 trivial steps
- The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In that case you are better off just doing the task directly.

## Task Fields

- **subject**: A brief, actionable title in imperative form (e.g., "Draft the quarterly summary outline")
- **description**: What needs to be done, including context and acceptance criteria
- **active_form** (optional): Present continuous form shown while the task is in_progress (e.g., "Drafting the quarterly summary outline"). If omitted, the subject is shown instead.

All tasks are created with status `pending`.

## Tips

- Create tasks with clear, specific subjects that describe the outcome
- Include enough detail in the description for another reader to understand and complete the task
- After creating tasks, use TaskUpdate to set up dependencies (add_blocked_by/add_blocks) if needed
- Check TaskList first to avoid creating duplicate tasks"""


def _task_create_description(team_active: bool = False) -> str:
    """Return the TaskCreate description, optionally with team tips."""
    base = _TASK_CREATE_DESCRIPTION
    if team_active:
        base = base.replace(
            "that require careful planning or multiple operations",
            "that require careful planning or multiple operations "
            "and potentially assigned to teammates",
        )
        base += (
            "\n\nTeam tips:\n"
            "- Include enough detail in the description "
            "for another agent to understand and complete the task\n"
            "- New tasks are created with status 'pending' and no owner "
            "- use TaskUpdate with the `owner` parameter to assign them"
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
