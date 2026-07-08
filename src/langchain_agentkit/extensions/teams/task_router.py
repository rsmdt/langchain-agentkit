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
    TaskUpdateError,
    apply_task_update,
    compute_blocks,
    filter_resolved_blockers,
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
    with contextlib.suppress(Exception):
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


def _op_update(
    op: dict[str, Any],
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    task_id = op.get("task_id")
    if not task_id:
        return _ack(request_id, error="Missing task_id"), tasks

    tasks = [dict(t) for t in tasks]
    try:
        task = apply_task_update(
            tasks,
            task_id,
            status=op.get("status"),
            subject=op.get("subject"),
            description=op.get("description"),
            active_form=op.get("active_form"),
            owner=op.get("owner"),
            metadata=op.get("metadata"),
            add_blocked_by=op.get("add_blocked_by"),
            add_blocks=op.get("add_blocks"),
            sender=op.get("sender"),
        )
    except TaskUpdateError as exc:
        return _ack(request_id, error=str(exc)), tasks

    # Completion nudge: tell teammate to check for next work
    extra: dict[str, Any] = {"task": task}
    if op.get("status") == "completed":
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
    result["blocks"] = compute_blocks(task_id, tasks)
    return _ack(request_id, task=result), tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ack(request_id: str, **fields: Any) -> str:
    """Build a JSON ack response with the given request_id and extra fields."""
    return json.dumps({"request_id": request_id, **fields})
