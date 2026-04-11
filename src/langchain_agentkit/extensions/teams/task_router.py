"""Router-side task operation processing.

The router node in :mod:`~langchain_agentkit.extensions.teams.extension`
uses these helpers to detect structured task operations from teammates
and apply them to the lead's ``state["tasks"]``.

Each operation mirrors the logic in
:mod:`~langchain_agentkit.extensions.tasks.tools` but operates on
plain lists instead of LangGraph ``Command`` objects.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from langchain_agentkit.extensions.tasks.core import (
    cascade_delete,
    filter_resolved_blockers,
    unresolved_blockers,
)
from langchain_agentkit.extensions.teams.task_proxy import TASK_OP_TYPE

if TYPE_CHECKING:
    from langchain_agentkit.extensions.teams.bus import TeamMessage, TeamMessageBus


async def _notify_assignment(
    bus: TeamMessageBus,
    result_json: str,
    old_owner: str | None,
    sender: str,
    timestamp: float,
) -> None:
    """Send task_assignment notification when owner changes (best-effort)."""
    result_parsed = json.loads(result_json)
    new_task = result_parsed.get("task")
    if not new_task or "error" in result_parsed:
        return
    new_owner = new_task.get("owner")
    if not new_owner or new_owner == old_owner:
        return
    notification = json.dumps(
        {
            "type": "task_assignment",
            "task_id": new_task["id"],
            "subject": new_task.get("subject", ""),
            "description": new_task.get("description", ""),
            "assigned_by": sender,
            "timestamp": str(timestamp),
        }
    )
    with contextlib.suppress(ValueError, Exception):
        await bus.send("lead", new_owner, notification)


async def classify_and_process(
    raw_messages: list[TeamMessage],
    tasks: list[dict[str, Any]],
    bus: TeamMessageBus,
) -> dict[str, Any]:
    """Classify drained messages into task ops vs regular, process both.

    Returns a state update dict with ``messages`` and/or ``tasks`` keys.
    """
    human_messages: list[Any] = []
    tasks_changed = False

    for m in raw_messages:
        parsed = try_parse_task_op(m.content)
        if parsed is not None:
            # Inject sender from message metadata (proxy payloads don't include it)
            parsed["sender"] = m.sender

            # Capture pre-update owner for assignment detection
            old_owner = None
            if parsed.get("op") == "update" and parsed.get("task_id"):
                existing = next((t for t in tasks if t["id"] == parsed["task_id"]), None)
                if existing:
                    old_owner = existing.get("owner")

            result, tasks = process_task_op(parsed, tasks)
            tasks_changed = True
            await bus.send("lead", m.sender, result)

            if parsed.get("op") == "update":
                await _notify_assignment(bus, result, old_owner, m.sender, m.timestamp)
        else:
            human_messages.append(HumanMessage(content=m.content))

    update: dict[str, Any] = {}
    if human_messages:
        update["messages"] = human_messages
    if tasks_changed:
        update["tasks"] = tasks
    return update


def try_parse_task_op(content: str) -> dict[str, Any] | None:
    """Attempt to parse a message as a task operation.

    Returns the parsed dict if it's a valid task_op message, else ``None``.
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None
    if parsed.get("type") != TASK_OP_TYPE:
        return None
    if "op" not in parsed or "request_id" not in parsed:
        return None
    return parsed


def process_task_op(
    op: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Process a task operation and return (ack_json, updated_tasks).

    The ack JSON always includes ``request_id`` so the caller can match
    it to the original request via :meth:`TeamMessageBus.request_response`.
    """
    request_id = op["request_id"]
    op_type = op["op"]

    if op_type == "create":
        return _op_create(op, tasks, request_id)
    if op_type == "update":
        return _op_update(op, tasks, request_id)
    if op_type == "list":
        return _op_list(tasks, request_id)
    if op_type == "get":
        return _op_get(op, tasks, request_id)

    return _ack(request_id, error=f"Unknown task operation: {op_type}"), tasks


# ---------------------------------------------------------------------------
# Operation handlers
# ---------------------------------------------------------------------------


def _op_create(
    op: dict[str, Any],
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    task: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "subject": op.get("subject", ""),
        "description": op.get("description", ""),
        "status": "pending",
        "active_form": op.get("active_form", ""),
    }
    tasks = [*tasks, task]
    return _ack(request_id, task=task), tasks


def _apply_scalar_fields(task: dict[str, Any], op: dict[str, Any]) -> None:
    """Apply simple scalar field updates from an op dict."""
    for field in ("status", "subject", "description", "active_form", "owner"):
        if op.get(field) is not None:
            task[field] = op[field]


def _apply_metadata(task: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Merge metadata into task, removing keys set to ``None``."""
    existing = dict(task.get("metadata") or {})
    for key, val in metadata.items():
        if val is None:
            existing.pop(key, None)
        else:
            existing[key] = val
    task["metadata"] = existing


def _apply_dependency_fields(
    task: dict[str, Any],
    tasks: list[dict[str, Any]],
    op: dict[str, Any],
) -> None:
    """Apply blocked_by/blocks dependency updates."""
    task_id = task["id"]
    if op.get("add_blocked_by"):
        task["blocked_by"] = list(
            dict.fromkeys(task.get("blocked_by", []) + op["add_blocked_by"]),
        )
    if op.get("add_blocks"):
        for target_id in op["add_blocks"]:
            target = next((t for t in tasks if t["id"] == target_id), None)
            if target is not None:
                target["blocked_by"] = list(
                    dict.fromkeys(target.get("blocked_by", []) + [task_id]),
                )


def _op_update(
    op: dict[str, Any],
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    task_id = op.get("task_id")
    if not task_id:
        return _ack(request_id, error="Missing task_id"), tasks

    tasks = [dict(t) for t in tasks]
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return _ack(request_id, error=f"Task '{task_id}' not found."), tasks

    new_status = op.get("status")
    new_owner = op.get("owner")
    sender = op.get("sender")

    # Dependency enforcement: block in_progress when blockers unresolved
    if new_status == "in_progress":
        unresolved = unresolved_blockers(task, tasks)
        if unresolved:
            ids = ", ".join(unresolved)
            msg = f"Task '{task_id}' is blocked by unresolved tasks: {ids}."
            return _ack(request_id, error=msg), tasks

    # Claim validation: prevent overwriting another agent's claim
    existing_owner = task.get("owner")
    if (
        new_owner
        and existing_owner
        and existing_owner != new_owner
        and task.get("status") == "in_progress"
    ):
        msg = f"Task '{task_id}' is already claimed by '{existing_owner}'."
        return _ack(request_id, error=msg), tasks

    # Deletion cascade: remove references from other tasks
    if new_status == "deleted":
        cascade_delete(task_id, tasks)

    # Auto-owner: when teammate sets in_progress without explicit owner
    if new_status == "in_progress" and not new_owner and not task.get("owner") and sender:
        op["owner"] = sender

    _apply_scalar_fields(task, op)
    if op.get("metadata") is not None:
        _apply_metadata(task, op["metadata"])
    _apply_dependency_fields(task, tasks, op)

    # Completion nudge: tell teammate to check for next work
    extra: dict[str, Any] = {"task": task}
    if new_status == "completed":
        extra["message"] = (
            "Task completed. Call TaskList now to find your next "
            "available task or see if your work unblocked others."
        )

    return _ack(request_id, **extra), tasks


def _op_list(
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
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
        filter_resolved_blockers(entry, completed_ids)
        summary.append(entry)
    return _ack(request_id, tasks=summary), tasks


def _op_get(
    op: dict[str, Any],
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    task_id = op.get("task_id")
    if not task_id:
        return _ack(request_id, error="Missing task_id"), tasks

    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        return _ack(request_id, error=f"Task '{task_id}' not found."), tasks

    result = dict(task)
    # Compute reverse blocks
    result["blocks"] = [
        t["id"]
        for t in tasks
        if task_id in t.get("blocked_by", []) and t.get("status") not in ("completed", "deleted")
    ]
    return _ack(request_id, task=result), tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ack(request_id: str, **fields: Any) -> str:
    """Build a JSON ack response with the given request_id and extra fields."""
    return json.dumps({"request_id": request_id, **fields})
