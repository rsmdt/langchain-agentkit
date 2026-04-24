"""Public domain types for the Tasks extension.

``Task`` and ``TaskStatus`` describe the shape of a task dict as managed
by the built-in Command-based task tools. They live here — not in
``tools/shared.py`` — because they are user-facing: consumers of the
tools' results inspect ``Task`` dicts, and handlers may annotate with
``TaskStatus`` when reading ``state["tasks"]``.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

TaskStatus = Literal["pending", "in_progress", "completed", "deleted"]


class _TaskOptional(TypedDict, total=False):
    blocked_by: list[str]
    blocks: list[str]
    owner: str
    metadata: dict[str, Any]


class Task(_TaskOptional):
    """Shape of a task dict managed by the task tools."""

    id: str
    subject: str
    description: str
    status: TaskStatus
    active_form: str


__all__ = ["Task", "TaskStatus"]
