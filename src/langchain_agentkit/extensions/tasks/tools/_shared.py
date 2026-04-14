"""Shared types, schemas, and helpers for task tools."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.tools import InjectedToolCallId, ToolException
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from langchain_agentkit.extensions.tasks.core import (
    cascade_delete,
    filter_resolved_blockers,
    unresolved_blockers,
)

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
    """Validate dependency and ownership constraints for status transitions."""
    if new_status == "in_progress":
        unresolved = _unresolved_blockers(task, tasks)
        if unresolved:
            ids = ", ".join(unresolved)
            raise ToolException(
                f"Task '{task['id']}' is blocked by unresolved tasks: {ids}. "
                "Complete blockers first or remove the dependency."
            )

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
