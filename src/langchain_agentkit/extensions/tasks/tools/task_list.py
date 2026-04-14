"""TaskList tool."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool

from langchain_agentkit.extensions.tasks.tools._shared import (
    _filter_resolved_blockers,
    _TaskListInput,
)

_TASK_LIST_DESCRIPTION = """List all tasks with their current status.

Use to:
- See what tasks are available to work on (status: pending, no owner, not blocked)
- Check overall progress on the work
- Find tasks that are blocked and need dependencies resolved
- Before creating tasks, to avoid duplicates
- After completing a task, to check for newly unblocked work

Prefer working on tasks in ID order (lowest ID first) when multiple are available, as earlier tasks often set up context for later ones.

Shared-queue workflow:
1. Call TaskList to find available work
2. Look for tasks with status `pending`, no owner, and empty blocked_by
3. Claim an available task using TaskUpdate (set owner to your name)"""


def _task_list(state: dict[str, Any]) -> str:
    """List all non-deleted, non-internal tasks with resolved blockers filtered."""
    tasks = state.get("tasks") or []
    completed_ids = {t["id"] for t in tasks if t.get("status") == "completed"}
    summary = []
    for t in tasks:
        if t.get("status") == "deleted":
            continue
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


def build_task_list_tool() -> Any:
    """Build the TaskList StructuredTool."""
    return StructuredTool.from_function(
        func=_task_list,
        name="TaskList",
        description=_TASK_LIST_DESCRIPTION,
        args_schema=_TaskListInput,
        handle_tool_error=True,
    )
