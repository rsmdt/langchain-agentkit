"""Router-side task operation processing.

The router node in :mod:`~langchain_agentkit.extensions.teams.extension`
uses these helpers to detect structured task operations from teammates
and apply them to the lead's ``state["tasks"]``.

Each operation mirrors the logic in
:mod:`~langchain_agentkit.extensions.tasks.tools` but operates on
plain lists instead of LangGraph ``Command`` objects.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage

from langchain_agentkit.extensions.teams.task_proxy import TASK_OP_TYPE

if TYPE_CHECKING:
    from langchain_agentkit.extensions.teams.bus import TeamMessage, TeamMessageBus


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
            result, tasks = process_task_op(parsed, tasks)
            tasks_changed = True
            await bus.send("lead", m.sender, result)
        else:
            human_messages.append(
                HumanMessage(
                    content=f"[Message from teammate '{m.sender}']: {m.content}",
                    additional_kwargs={
                        "sender": m.sender,
                        "type": "teammate_message",
                    },
                )
            )

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

    _apply_scalar_fields(task, op)
    if op.get("metadata") is not None:
        _apply_metadata(task, op["metadata"])
    _apply_dependency_fields(task, tasks, op)

    return _ack(request_id, task=task), tasks


def _op_list(
    tasks: list[dict[str, Any]],
    request_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    summary = [
        {
            "id": t["id"],
            "subject": t.get("subject", ""),
            "status": t.get("status", "pending"),
            "owner": t.get("owner", ""),
            "blocked_by": t.get("blocked_by", []),
        }
        for t in tasks
        if t.get("status") != "deleted"
    ]
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
        if task_id in t.get("blocked_by", [])
        and t.get("status") not in ("completed", "deleted")
    ]
    return _ack(request_id, task=result), tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ack(request_id: str, **fields: Any) -> str:
    """Build a JSON ack response with the given request_id and extra fields."""
    return json.dumps({"request_id": request_id, **fields})
