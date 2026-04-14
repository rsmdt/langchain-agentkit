"""Tasks tools package.

Command-based task management tools for LangGraph agents. Tools use
``InjectedState`` to read current tasks from graph state and return
``Command(update={"tasks": ...})`` to apply changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_agentkit.extensions.tasks.tools._shared import (
    Task,
    TaskStatus,
)
from langchain_agentkit.extensions.tasks.tools.task_create import build_task_create_tool
from langchain_agentkit.extensions.tasks.tools.task_get import build_task_get_tool
from langchain_agentkit.extensions.tasks.tools.task_list import build_task_list_tool
from langchain_agentkit.extensions.tasks.tools.task_stop import build_task_stop_tool
from langchain_agentkit.extensions.tasks.tools.task_update import build_task_update_tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def create_task_tools(*, team_active: bool = False) -> list[BaseTool]:
    """Create Command-based task management tools.

    Returns five tools: TaskCreate, TaskUpdate, TaskList, TaskGet, TaskStop.

    Args:
        team_active: When True, TaskCreate description includes team tips.
    """
    return [
        build_task_create_tool(team_active),
        build_task_update_tool(),
        build_task_list_tool(),
        build_task_get_tool(),
        build_task_stop_tool(),
    ]


__all__ = [
    "Task",
    "TaskStatus",
    "create_task_tools",
]
