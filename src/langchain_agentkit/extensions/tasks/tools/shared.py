"""Shared schemas and helpers for task tools.

Public domain types ``Task`` and ``TaskStatus`` live in
:mod:`langchain_agentkit.extensions.tasks.types`. They are re-imported
here for backward-compatible intra-package use.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from langchain_agentkit.extensions.tasks.types import Task, TaskStatus

__all__ = ["Task", "TaskStatus"]


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
