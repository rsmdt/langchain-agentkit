"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

_LIST_KEYS = ("blocked_by", "blocks")


def _merge_task_pair(
    existing: dict[str, Any], incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge two versions of the same task (same ID).

    - List fields (``blocked_by``, ``blocks``) are unioned and deduplicated.
    - ``metadata`` dicts are merged (incoming keys win, ``None`` deletes).
    - All other fields take the incoming value.
    """
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
    left: list[dict[str, Any]], right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer that merges task lists by ID.

    Handles concurrent ``Command`` updates from parallel tool calls
    (e.g., multiple ``TaskCreate`` in one step).

    - New tasks (ID not in left) are appended.
    - Existing tasks are deep-merged: list fields (``blocked_by``,
      ``blocks``) are unioned, ``metadata`` is merged, scalar fields
      take the incoming value.
    """
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


class AgentState(TypedDict, total=False):
    """Minimal state for multi-agent graphs.

    Extend with your own fields::

        class MyState(AgentState):
            my_custom_field: str

    Fields:
        messages: Conversation history with LangGraph message aggregation.
        sender: Name of the last node that called tools (for routing back).
        tasks: Task list managed by TasksMiddleware tools.
    """

    messages: Annotated[list[Any], add_messages]
    sender: str
    tasks: Annotated[list[dict[str, Any]], _merge_tasks]
