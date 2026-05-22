"""TaskGet tool."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool, ToolException

from langchain_agentkit.extensions.tasks.tools.shared import (
    _compute_blocks,
    _TaskGetInput,
)

_TASK_GET_DESCRIPTION = """Get full details of a task by ID — subject, description, status, owner, blocks, blocked_by, metadata.

Use before starting work to read full requirements and dependencies. Verify blocked_by is empty before beginning."""


def _task_get(task_id: str, state: dict[str, Any]) -> str:
    """Get full details of a task by ID."""
    tasks = state.get("tasks") or []
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        raise ToolException(f"Task '{task_id}' not found.")
    result = dict(task)
    result["blocks"] = _compute_blocks(task_id, tasks)
    return json.dumps(result)


def build_task_get_tool() -> Any:
    """Build the TaskGet StructuredTool."""
    return StructuredTool.from_function(
        func=_task_get,
        name="TaskGet",
        description=_TASK_GET_DESCRIPTION,
        args_schema=_TaskGetInput,
        handle_tool_error=True,
    )
