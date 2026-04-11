"""Pure task-list operations shared by tools and the team router.

These functions operate on plain ``list[dict]`` task collections with
no LangGraph or framework dependencies.
"""

from __future__ import annotations

from typing import Any


def unresolved_blockers(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> list[str]:
    """Return IDs of blockers that are NOT yet completed."""
    blocked_by = task.get("blocked_by") or []
    if not blocked_by:
        return []
    completed = {t["id"] for t in tasks if t.get("status") == "completed"}
    return [bid for bid in blocked_by if bid not in completed]


def cascade_delete(task_id: str, tasks: list[dict[str, Any]]) -> None:
    """Remove *task_id* from all other tasks' blocks/blocked_by lists."""
    for t in tasks:
        if t["id"] == task_id:
            continue
        blocked_by = t.get("blocked_by")
        if blocked_by and task_id in blocked_by:
            t["blocked_by"] = [bid for bid in blocked_by if bid != task_id]
        blocks = t.get("blocks")
        if blocks and task_id in blocks:
            t["blocks"] = [bid for bid in blocks if bid != task_id]


def filter_resolved_blockers(
    task_summary: dict[str, Any],
    completed_ids: set[str],
) -> None:
    """Remove completed task IDs from blocked_by in a task summary."""
    blocked_by = task_summary.get("blocked_by") or []
    if blocked_by:
        task_summary["blocked_by"] = [bid for bid in blocked_by if bid not in completed_ids]
