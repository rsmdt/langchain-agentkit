"""State schema and reducers for task management."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

_LIST_KEYS = ("blocked_by", "blocks")


def _merge_task_pair(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge two versions of the same task (same ID)."""
    merged = dict(existing)
    for key, val in incoming.items():
        if key in _LIST_KEYS:
            old = merged.get(key, []) or []
            merged[key] = list(dict.fromkeys(old + (val or [])))
        elif key == "metadata":
            old_meta = dict(merged.get("metadata") or {})
            for mk, mv in (val or {}).items():
                if mv is None:
                    old_meta.pop(mk, None)
                else:
                    old_meta[mk] = mv
            merged["metadata"] = old_meta
        else:
            merged[key] = val
    return merged


def _merge_tasks(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer that merges task lists by ID."""
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for task in left or []:
        tid = task.get("id", "")
        if tid:
            by_id[tid] = dict(task)
            order.append(tid)
    for task in right or []:
        tid = task.get("id", "")
        if not tid:
            continue
        if tid in by_id:
            by_id[tid] = _merge_task_pair(by_id[tid], task)
        else:
            by_id[tid] = dict(task)
            order.append(tid)
    return [by_id[tid] for tid in order if tid in by_id]


class TasksState(TypedDict, total=False):
    """State mixin for task management."""

    tasks: Annotated[list[dict[str, Any]], _merge_tasks]
