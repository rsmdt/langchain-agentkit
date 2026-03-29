"""TasksExtension — task management tools and system prompt guidance.

Usage::

    from langchain_agentkit import TasksExtension

    # Default — auto-creates Command-based task tools
    mw = TasksExtension()
    mw.tools   # [TaskCreate, TaskUpdate, TaskList, TaskGet]
    mw.prompt(state, runtime)  # Base agent prompt + task context

    # Custom tools
    mw = TasksExtension(task_tools=[my_task_create, my_task_update])

    # Custom formatter
    mw = TasksExtension(formatter=my_format_function)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_base_agent_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "base_agent.md")
_task_management_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "task_management.md")

BASE_AGENT_PROMPT = _base_agent_prompt.format()
TASK_MANAGEMENT_PROMPT = _task_management_prompt.format()

_STATUS_ICONS = {
    "completed": "x",
    "in_progress": ">",
    "deleted": "-",
}


def format_task_context(tasks: list[dict[str, Any]]) -> str:
    """Format active tasks into a rich prompt section.

    Features:
        - Status icons: ``[x]`` completed, ``[>]`` in-progress, ``[-]`` deleted, ``[ ]`` pending
        - Filters out deleted tasks
        - Shows active form text for in-progress tasks
        - Shows blocked-by dependencies for pending tasks
        - Numbered list with XML wrapping

    Returns ``TASK_MANAGEMENT_PROMPT`` when no visible tasks remain.

    Args:
        tasks: List of task dicts with at least 'subject' and 'status' keys.
    """
    visible = [t for t in tasks if t.get("status") != "deleted"]
    if not visible:
        return TASK_MANAGEMENT_PROMPT

    lines = []
    for i, task in enumerate(visible, 1):
        status = task.get("status", "pending")
        icon = _STATUS_ICONS.get(status, " ")
        subject = task.get("subject", "")
        suffix = ""
        if status == "in_progress" and task.get("active_form"):
            suffix = f' -- "{task["active_form"]}"'
        blocked_by = task.get("blocked_by", [])
        if blocked_by and status == "pending":
            dep_str = ", ".join(blocked_by)
            suffix = f" (pending, blocked by: {dep_str})"
        lines.append(f"{i}. [{icon}] {subject}{suffix}")

    body = "\n".join(lines)
    return (
        f"## Current Tasks\n\n<tasks>\n{body}\n</tasks>\n\n"
        "Review your current tasks before proceeding. Update task status as you work."
    )


class TasksExtension(Extension):
    """Extension providing task management tools and system prompt guidance.

    Tools: Uses Command-based task tools by default. Accepts custom tools
    via constructor for override.

    Prompt: Base agent behavior + task context (rich formatting with status
    icons, dependency tracking, and XML wrapping) or task management
    guidance when no tasks exist.

    Args:
        task_tools: Optional custom task tools. If ``None``, creates
            default Command-based tools via ``create_task_tools()``.
        formatter: Optional custom function to format tasks into a prompt
            section. Receives ``list[dict]`` and returns ``str``.
            Defaults to ``format_task_context``.

    Example::

        # Default — auto-creates Command-based task tools
        mw = TasksExtension()

        # Custom tools
        mw = TasksExtension(task_tools=[my_create, my_update])

        # Custom formatter
        mw = TasksExtension(formatter=my_format_function)
    """

    def __init__(
        self,
        task_tools: list[BaseTool] | None = None,
        formatter: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        if task_tools is not None:
            self._tools = tuple(task_tools)
        else:
            from langchain_agentkit.tools.task import create_task_tools

            self._tools = tuple(create_task_tools())
        self._formatter = formatter or format_task_context

    @property
    def state_schema(self) -> type:
        """Tasks require ``TasksState`` in the graph state."""
        from langchain_agentkit.state import TasksState

        return TasksState

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        sections = [BASE_AGENT_PROMPT]
        tasks = state.get("tasks") or []
        sections.append(self._formatter(tasks))
        return "\n\n".join(sections)

