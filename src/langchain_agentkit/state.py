"""Composable state schemas for LangGraph agents.

``AgentKitState`` is the minimal base — just messages and sender.
Middleware adds state keys via mixins (e.g., ``TasksState``).

Usage::

    from langchain_agentkit.state import AgentKitState, TasksState

    # Compose manually
    class MyState(AgentKitState, TasksState):
        my_field: str

    # Or let AgentKit compose from middleware automatically
    kit = AgentKit([TasksMiddleware()])
    kit.state_schema  # → composed TypedDict with messages + tasks
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Task list reducer
# ---------------------------------------------------------------------------

_LIST_KEYS = ("blocked_by", "blocks")


def _merge_task_pair(
    existing: dict[str, Any],
    incoming: dict[str, Any],
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
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
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


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class AgentKitState(TypedDict, total=False):
    """Minimal base state — always present in any agentkit graph.

    Contains only the fields required for the ReAct loop to function.
    Middleware adds additional keys via mixin TypedDicts.
    """

    messages: Annotated[list[Any], add_messages]
    sender: str


class TasksState(TypedDict, total=False):
    """State mixin for task management.

    Added to the graph state when ``TasksMiddleware`` is used.
    """

    tasks: Annotated[list[dict[str, Any]], _merge_tasks]


# ---------------------------------------------------------------------------
# Delegation state
# ---------------------------------------------------------------------------


def _merge_delegation_log(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append-only reducer for delegation log entries."""
    return (left or []) + (right or [])


class SubAgentState(TypedDict, total=False):
    """State mixin for agent delegation.

    Added to the graph state when ``AgentMiddleware`` is used.
    """

    delegation_log: Annotated[list[dict[str, Any]], _merge_delegation_log]


# ---------------------------------------------------------------------------
# Team state
# ---------------------------------------------------------------------------


def _merge_team_members(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge team members by name — latest update wins per member."""
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for member in left or []:
        n = member.get("name", "")
        if n:
            by_name[n] = dict(member)
            order.append(n)
    for member in right or []:
        n = member.get("name", "")
        if not n:
            continue
        if n in by_name:
            by_name[n].update(member)
        else:
            by_name[n] = dict(member)
            order.append(n)
    return [by_name[n] for n in order if n in by_name]


def _append_messages(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append-only reducer for team messages."""
    return (left or []) + (right or [])


class TeamState(TypedDict, total=False):
    """State mixin for team coordination.

    Added to the graph state when ``AgentTeamMiddleware`` is used.
    """

    team_members: Annotated[list[dict[str, Any]], _merge_team_members]
    team_messages: Annotated[list[dict[str, Any]], _append_messages]
    team_name: str | None
