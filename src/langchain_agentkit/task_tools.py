"""Command-based task management tools for LangGraph agents.

Tools use ``InjectedState`` to read current tasks from graph state and
return ``Command(update={"tasks": ...})`` to apply changes. This makes
them compatible with LangGraph's ``ToolNode`` out of the box.

Usage::

    from langchain_agentkit import create_task_tools, TasksMiddleware

    # Default — TasksMiddleware creates tools automatically
    mw = TasksMiddleware()

    # Explicit — pass tools to middleware
    mw = TasksMiddleware(task_tools=create_task_tools())

    # Custom state key
    mw = TasksMiddleware(task_tools=create_task_tools(state_key="my_tasks"))
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class _TaskCreateInput(BaseModel):
    subject: str = Field(description='Imperative title, e.g. "Analyze problem context".')
    description: str = Field(description="Detailed requirements, context, and acceptance criteria.")
    active_form: str = Field(
        default="",
        description='Spinner text shown when in progress, e.g. "Analyzing problem context".',
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional arbitrary key-value data.",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TaskUpdateInput(BaseModel):
    task_id: str = Field(description="Task ID to update.")
    status: str | None = Field(
        default=None,
        description="New status -- pending, in_progress, completed, or deleted.",
    )
    subject: str | None = Field(default=None, description="Updated imperative title.")
    description: str | None = Field(default=None, description="Updated requirements/context.")
    active_form: str | None = Field(default=None, description="Updated spinner text.")
    owner: str | None = Field(default=None, description="Specialist slug claiming this task.")
    add_blocks: list[str] | None = Field(default=None, description="Task IDs this task now blocks.")
    add_blocked_by: list[str] | None = Field(
        default=None, description="Task IDs now blocking this task."
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Keys to merge (set value to null to delete a key).",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TaskListInput(BaseModel):
    state: Annotated[dict[str, Any], InjectedState]


class _TaskGetInput(BaseModel):
    task_id: str = Field(description="Task ID to retrieve.")
    state: Annotated[dict[str, Any], InjectedState]


def _task_create(
    subject: str,
    description: str,
    state: dict[str, Any],
    tool_call_id: str,
    active_form: str = "",
    metadata: dict[str, Any] | None = None,
) -> Command:  # type: ignore[type-arg]
    """Create a new task with pending status."""
    task: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "subject": subject,
        "description": description,
        "status": "pending",
        "active_form": active_form,
        "owner": None,
        "blocks": [],
        "blocked_by": [],
        "metadata": metadata or {},
        "created_at": datetime.now(UTC).isoformat(),
    }
    tasks = list(state.get("tasks") or [])
    tasks.append(task)
    return Command(
        update={
            "tasks": tasks,
            "messages": [ToolMessage(content=json.dumps(task), tool_call_id=tool_call_id)],
        }
    )


def _task_update(
    task_id: str,
    state: dict[str, Any],
    tool_call_id: str,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    owner: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Command:  # type: ignore[type-arg]
    """Update an existing task."""
    tasks = [dict(t) for t in (state.get("tasks") or [])]
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")

    if status is not None:
        task["status"] = status
    if subject is not None:
        task["subject"] = subject
    if description is not None:
        task["description"] = description
    if active_form is not None:
        task["active_form"] = active_form
    if owner is not None:
        task["owner"] = owner
    if add_blocks:
        task["blocks"] = list(dict.fromkeys(task.get("blocks", []) + add_blocks))
    if add_blocked_by:
        task["blocked_by"] = list(dict.fromkeys(task.get("blocked_by", []) + add_blocked_by))
    if metadata:
        existing = task.get("metadata") or {}
        merged = {**existing, **metadata}
        task["metadata"] = {k: v for k, v in merged.items() if v is not None}

    return Command(
        update={
            "tasks": tasks,
            "messages": [ToolMessage(content=json.dumps(task), tool_call_id=tool_call_id)],
        }
    )


def _task_list(state: dict[str, Any]) -> str:
    """List all non-deleted tasks."""
    tasks = state.get("tasks") or []
    summary = [
        {
            "id": t["id"],
            "subject": t.get("subject", ""),
            "status": t.get("status", "pending"),
            "owner": t.get("owner"),
            "blocked_by": t.get("blocked_by", []),
        }
        for t in tasks
        if t.get("status") != "deleted"
    ]
    return json.dumps(summary)


def _task_get(task_id: str, state: dict[str, Any]) -> str:
    """Get full details of a task by ID."""
    tasks = state.get("tasks") or []
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")
    return json.dumps(task)


def create_task_tools() -> list[BaseTool]:
    """Create Command-based task management tools.

    Returns four tools: TaskCreate, TaskUpdate, TaskList, TaskGet.

    Each tool uses ``InjectedState`` to read tasks from graph state and
    returns ``Command(update={"tasks": ...})`` (or a JSON string for
    read-only operations) to update state via LangGraph's ``ToolNode``.

    Example::

        from langchain_agentkit import create_task_tools, TasksMiddleware

        mw = TasksMiddleware(task_tools=create_task_tools())
    """
    task_create = StructuredTool.from_function(
        func=_task_create,
        name="TaskCreate",
        description=(
            "Create a new task with pending status. Use for multi-step work (3+ steps), "
            "complex tasks, or when a user gives a list of things to do. "
            "Do NOT use for single trivial tasks or conversational questions. "
            "Break down complex tasks into smaller, actionable steps."
        ),
        args_schema=_TaskCreateInput,
        handle_tool_error=True,
    )

    task_update = StructuredTool.from_function(
        func=_task_update,
        name="TaskUpdate",
        description=(
            "Update an existing task. Set status to in_progress BEFORE starting work. "
            "Set completed ONLY when FULLY done and verified. Set deleted to remove. "
            "Never mark completed if work is partial or errors are unresolved."
        ),
        args_schema=_TaskUpdateInput,
        handle_tool_error=True,
    )

    task_list = StructuredTool.from_function(
        func=_task_list,
        name="TaskList",
        description=(
            "List all tasks with their current status. Use to check progress, "
            "find available work, or before creating tasks to avoid duplicates. "
            "Prefer working lowest-created-first (earlier tasks set up context)."
        ),
        args_schema=_TaskListInput,
        handle_tool_error=True,
    )

    task_get = StructuredTool.from_function(
        func=_task_get,
        name="TaskGet",
        description=(
            "Get full details of a task by ID. Use before starting a task to read "
            "requirements, or to check dependency chains."
        ),
        args_schema=_TaskGetInput,
        handle_tool_error=True,
    )

    return [task_create, task_update, task_list, task_get]
