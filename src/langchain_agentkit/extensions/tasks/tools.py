"""Command-based task management tools for LangGraph agents.

Tools use ``InjectedState`` to read current tasks from graph state and
return ``Command(update={"tasks": ...})`` to apply changes. This makes
them compatible with LangGraph's ``ToolNode`` out of the box.

Usage::

    from langchain_agentkit import create_task_tools, TasksExtension

    # Default — TasksExtension creates tools automatically
    mw = TasksExtension()

    # Explicit — pass tools to extension
    mw = TasksExtension(task_tools=create_task_tools())
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool, ToolException
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from langchain_agentkit.extensions.tasks.core import (
    cascade_delete,
    filter_resolved_blockers,
    unresolved_blockers,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

TaskStatus = Literal["pending", "in_progress", "completed", "deleted"]


class _TaskOptional(TypedDict, total=False):
    blocked_by: list[str]
    blocks: list[str]
    owner: str
    metadata: dict[str, Any]


class Task(_TaskOptional):
    """Shape of a task dict managed by the task tools."""

    id: str
    subject: str
    description: str
    status: TaskStatus
    active_form: str


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _TaskCreateInput(BaseModel):
    subject: str = Field(
        description=(
            "A brief, actionable title in imperative form "
            '(e.g., "Fix authentication bug in login flow").'
        ),
    )
    description: str = Field(
        description=(
            "Detailed description of what needs to be done, "
            "including context and acceptance criteria."
        ),
    )
    active_form: str = Field(
        default="",
        description=(
            "Present continuous form shown in the spinner when the task is "
            'in_progress (e.g., "Fixing authentication bug"). '
            "If omitted, the spinner shows the subject instead."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TaskUpdateInput(BaseModel):
    task_id: str = Field(description="Task ID to update.")
    status: str | None = Field(
        default=None,
        description="New status: pending, in_progress, completed, or deleted.",
    )
    subject: str | None = Field(
        default=None,
        description='Updated imperative title (e.g., "Run tests").',
    )
    description: str | None = Field(
        default=None,
        description="Updated requirements/context.",
    )
    active_form: str | None = Field(
        default=None,
        description=(
            'Present continuous form shown in spinner when in_progress (e.g., "Running tests").'
        ),
    )
    owner: str | None = Field(
        default=None,
        description="Set the task owner (agent name).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=("Metadata keys to merge into the task. Set a key to null to delete it."),
    )
    add_blocked_by: list[str] | None = Field(
        default=None,
        description="Task IDs that must complete before this one can start.",
    )
    add_blocks: list[str] | None = Field(
        default=None,
        description="Task IDs that cannot start until this one completes.",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TaskListInput(BaseModel):
    state: Annotated[dict[str, Any], InjectedState]


class _TaskGetInput(BaseModel):
    task_id: str = Field(description="Task ID to retrieve.")
    state: Annotated[dict[str, Any], InjectedState]


class _TaskStopInput(BaseModel):
    task_id: str = Field(description="ID of the task to stop.")
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_blocks(task_id: str, tasks: list[dict[str, Any]]) -> list[str]:
    """Compute which tasks are blocked by *task_id* (reverse lookup)."""
    return [
        t["id"]
        for t in tasks
        if task_id in t.get("blocked_by", []) and t.get("status") not in ("completed", "deleted")
    ]


_unresolved_blockers = unresolved_blockers
_cascade_delete = cascade_delete
_filter_resolved_blockers = filter_resolved_blockers


def _validate_claim(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
    new_status: str | None,
    new_owner: str | None,
) -> None:
    """Validate dependency and ownership constraints for status transitions.

    Raises ``ToolException`` if:
    - Transitioning to ``in_progress`` with unresolved blockers.
    - Setting owner on a task already claimed by a different agent.
    """
    # Dependency enforcement: block in_progress when blockers unresolved
    if new_status == "in_progress":
        unresolved = _unresolved_blockers(task, tasks)
        if unresolved:
            ids = ", ".join(unresolved)
            raise ToolException(
                f"Task '{task['id']}' is blocked by unresolved tasks: {ids}. "
                "Complete blockers first or remove the dependency."
            )

    # Claim validation: prevent overwriting another agent's claim
    existing_owner = task.get("owner")
    if (
        new_owner
        and existing_owner
        and existing_owner != new_owner
        and task.get("status") == "in_progress"
    ):
        raise ToolException(
            f"Task '{task['id']}' is already claimed by '{existing_owner}'. "
            "Wait for them to finish or use TaskStop to release it."
        )


_cascade_delete = cascade_delete
_filter_resolved_blockers = filter_resolved_blockers


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


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


def _merge_metadata(
    task: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Merge metadata into task, removing keys set to None."""
    existing = dict(task.get("metadata") or {})
    for key, val in metadata.items():
        if val is None:
            existing.pop(key, None)
        else:
            existing[key] = val
    task["metadata"] = existing


def _apply_blocks(
    task_id: str,
    tasks: list[dict[str, Any]],
    target_ids: list[str],
) -> None:
    """For each target, add *task_id* to its blocked_by list."""
    for target_id in target_ids:
        target = next((t for t in tasks if t["id"] == target_id), None)
        if target is not None:
            target["blocked_by"] = list(
                dict.fromkeys(target.get("blocked_by", []) + [task_id]),
            )


def _apply_task_fields(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    owner: str | None = None,
    metadata: dict[str, Any] | None = None,
    add_blocked_by: list[str] | None = None,
    add_blocks: list[str] | None = None,
) -> None:
    """Apply field updates to a task dict in place."""
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
    if metadata is not None:
        _merge_metadata(task, metadata)
    if add_blocked_by:
        task["blocked_by"] = list(
            dict.fromkeys(task.get("blocked_by", []) + add_blocked_by),
        )
    if add_blocks:
        _apply_blocks(task["id"], tasks, add_blocks)


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


def _task_list(state: dict[str, Any]) -> str:
    """List all non-deleted, non-internal tasks with resolved blockers filtered."""
    tasks = state.get("tasks") or []
    completed_ids = {t["id"] for t in tasks if t.get("status") == "completed"}
    summary = []
    for t in tasks:
        if t.get("status") == "deleted":
            continue
        # Hide internal tasks
        if (t.get("metadata") or {}).get("_internal") is True:
            continue
        entry = {
            "id": t["id"],
            "subject": t.get("subject", ""),
            "status": t.get("status", "pending"),
            "owner": t.get("owner", ""),
            "blocked_by": t.get("blocked_by", []),
        }
        _filter_resolved_blockers(entry, completed_ids)
        summary.append(entry)
    return json.dumps(summary)


def _task_get(task_id: str, state: dict[str, Any]) -> str:
    """Get full details of a task by ID."""
    tasks = state.get("tasks") or []
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")
    # Include computed blocks (reverse of blocked_by)
    result = dict(task)
    result["blocks"] = _compute_blocks(task_id, tasks)
    return json.dumps(result)


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


# ---------------------------------------------------------------------------
# Tool descriptions (from spec)
# ---------------------------------------------------------------------------

_TASK_CREATE_DESCRIPTION = """\
Create a new task with pending status.

Use this tool proactively in these scenarios:
- Complex multi-step tasks requiring 3+ distinct steps or actions
- Non-trivial tasks that require careful planning or multiple operations
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated)
- After receiving new instructions, to capture requirements as tasks

Do NOT use when:
- There is only a single, straightforward task
- The task is trivial and tracking it provides no organizational benefit
- The task can be completed in less than 3 trivial steps
- The task is purely conversational or informational

Tips:
- Create tasks with clear, specific subjects that describe the outcome
- Include enough detail in the description for another agent to understand and complete the task
- After creating tasks, use TaskUpdate to set up dependencies (addBlockedBy/addBlocks) if needed
- Check TaskList first to avoid creating duplicate tasks\
"""

_TASK_UPDATE_DESCRIPTION = """\
Update an existing task.

Mark tasks as resolved:
- When you have completed the work described in a task
- IMPORTANT: Always mark assigned tasks as resolved when finished
- After resolving, call TaskList to find your next task
- ONLY mark completed when FULLY accomplished
- Never mark completed if tests are failing, implementation is partial, or errors are unresolved
- If blocked, keep as in_progress and create a new task describing what needs resolution

Delete tasks:
- When a task is no longer relevant or was created in error
- Setting status to 'deleted' permanently removes the task

Update task details:
- When requirements change or become clearer
- When establishing dependencies between tasks

Fields: status, subject, description, activeForm, owner, metadata, addBlocks, addBlockedBy

Status workflow: pending → in_progress → completed. Use 'deleted' to permanently remove.\
"""

_TASK_LIST_DESCRIPTION = """\
List all tasks with their current status.

Use to:
- See what tasks are available to work on (status: pending, no owner, not blocked)
- Check overall progress on the project
- Find tasks that are blocked and need dependencies resolved
- Before creating tasks, to avoid duplicates
- After completing a task, to check for newly unblocked work

Prefer working on tasks in ID order (lowest ID first) when multiple are available, \
as earlier tasks often set up context for later ones.

Teammate workflow:
1. Call TaskList to find available work
2. Look for tasks with status 'pending', no owner, and empty blockedBy
3. Claim an available task using TaskUpdate (set owner to your name)\
"""

_TASK_GET_DESCRIPTION = """\
Get full details of a task by ID.

Use when:
- You need the full description and context before starting work on a task
- To understand task dependencies (what it blocks, what blocks it)
- After being assigned a task, to get complete requirements

Returns: subject, description, status, owner, blocks, blockedBy, metadata.

Tip: After fetching a task, verify its blockedBy list is empty before beginning work.\
"""

_TASK_STOP_DESCRIPTION = """\
Stop a running background task by its ID.

Sets the task status back to pending. Use when you need to terminate a \
long-running task. Only works on tasks with status 'in_progress'.\
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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


def create_task_tools(*, team_active: bool = False) -> list[BaseTool]:
    """Create Command-based task management tools.

    Returns five tools: TaskCreate, TaskUpdate, TaskList, TaskGet, TaskStop.

    Each tool uses ``InjectedState`` to read tasks from graph state and
    returns ``Command(update={"tasks": ...})`` (or a JSON string for
    read-only operations) to update state via LangGraph's ``ToolNode``.

    Args:
        team_active: When True, TaskCreate description includes team tips.

    Example::

        from langchain_agentkit import create_task_tools, TasksExtension

        mw = TasksExtension(task_tools=create_task_tools())
    """
    task_create = StructuredTool.from_function(
        func=_task_create,
        name="TaskCreate",
        description=_task_create_description(team_active),
        args_schema=_TaskCreateInput,
        handle_tool_error=True,
    )

    task_update = StructuredTool.from_function(
        func=_task_update,
        name="TaskUpdate",
        description=_TASK_UPDATE_DESCRIPTION,
        args_schema=_TaskUpdateInput,
        handle_tool_error=True,
    )

    task_list = StructuredTool.from_function(
        func=_task_list,
        name="TaskList",
        description=_TASK_LIST_DESCRIPTION,
        args_schema=_TaskListInput,
        handle_tool_error=True,
    )

    task_get = StructuredTool.from_function(
        func=_task_get,
        name="TaskGet",
        description=_TASK_GET_DESCRIPTION,
        args_schema=_TaskGetInput,
        handle_tool_error=True,
    )

    task_stop = StructuredTool.from_function(
        func=_task_stop,
        name="TaskStop",
        description=_TASK_STOP_DESCRIPTION,
        args_schema=_TaskStopInput,
        handle_tool_error=True,
    )

    return [task_create, task_update, task_list, task_get, task_stop]
