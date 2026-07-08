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


def compute_blocks(task_id: str, tasks: list[dict[str, Any]]) -> list[str]:
    """Compute which open tasks are blocked by *task_id* (reverse lookup)."""
    return [
        t["id"]
        for t in tasks
        if task_id in t.get("blocked_by", []) and t.get("status") not in ("completed", "deleted")
    ]


class TaskUpdateError(ValueError):
    """A task update violated a transition constraint."""


def _merge_metadata(task: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Merge metadata into task, removing keys set to ``None``."""
    existing = dict(task.get("metadata") or {})
    for key, val in metadata.items():
        if val is None:
            existing.pop(key, None)
        else:
            existing[key] = val
    task["metadata"] = existing


def _apply_blocks(task_id: str, tasks: list[dict[str, Any]], target_ids: list[str]) -> None:
    """For each target, add *task_id* to its blocked_by list."""
    for target_id in target_ids:
        target = next((t for t in tasks if t["id"] == target_id), None)
        if target is not None:
            target["blocked_by"] = list(
                dict.fromkeys(target.get("blocked_by", []) + [task_id]),
            )


def _validate_transition(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
    status: str | None,
    owner: str | None,
) -> None:
    """Enforce blocker and claim constraints; raise ``TaskUpdateError``."""
    task_id = task["id"]
    if status == "in_progress":
        unresolved = unresolved_blockers(task, tasks)
        if unresolved:
            ids = ", ".join(unresolved)
            raise TaskUpdateError(
                f"Task '{task_id}' is blocked by unresolved tasks: {ids}. "
                "Complete blockers first or remove the dependency."
            )

    existing_owner = task.get("owner")
    if owner and existing_owner and existing_owner != owner and task.get("status") == "in_progress":
        raise TaskUpdateError(
            f"Task '{task_id}' is already claimed by '{existing_owner}'. "
            "Wait for them to finish, or set its status back to pending to release it."
        )


def _apply_scalar_fields(task: dict[str, Any], fields: dict[str, Any]) -> None:
    """Set the scalar fields that are not ``None``."""
    for key, value in fields.items():
        if value is not None:
            task[key] = value


def apply_task_update(
    tasks: list[dict[str, Any]],
    task_id: str,
    *,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    owner: str | None = None,
    metadata: dict[str, Any] | None = None,
    add_blocked_by: list[str] | None = None,
    add_blocks: list[str] | None = None,
    sender: str | None = None,
) -> dict[str, Any]:
    """Validate and apply one task update in place; return the updated task.

    The single source of truth for update semantics — the ``TaskUpdate``
    tool and the team router both delegate here, so a transition yields
    the same state no matter which path carried it.

    ``sender`` identifies the requesting agent when known. An identified
    agent starting an unowned task without naming an owner auto-claims
    it, so other agents see the claim and cannot double-claim.

    Raises:
        TaskUpdateError: Unknown ``task_id``, unresolved blockers on an
            ``in_progress`` transition, or a claim conflict.
    """
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise TaskUpdateError(f"Task '{task_id}' not found.")

    _validate_transition(task, tasks, status, owner)

    if status == "deleted":
        cascade_delete(task_id, tasks)

    if status == "in_progress" and not owner and not task.get("owner") and sender:
        owner = sender

    _apply_scalar_fields(
        task,
        {
            "status": status,
            "subject": subject,
            "description": description,
            "active_form": active_form,
            "owner": owner,
        },
    )
    if metadata is not None:
        _merge_metadata(task, metadata)
    if add_blocked_by:
        task["blocked_by"] = list(
            dict.fromkeys(task.get("blocked_by", []) + add_blocked_by),
        )
    if add_blocks:
        _apply_blocks(task_id, tasks, add_blocks)

    return task
