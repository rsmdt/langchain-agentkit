"""TasksExtension — task management tools and system prompt guidance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent

_base_agent_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "base_agent_prompt.md")
_task_management_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "prompt.md")

BASE_AGENT_PROMPT = _base_agent_prompt.format()
TASK_MANAGEMENT_PROMPT = _task_management_prompt.format()

_STATUS_ICONS = {
    "completed": "x",
    "in_progress": ">",
    "deleted": "-",
}


def format_task_context(tasks: list[dict[str, Any]]) -> str:
    """Format active tasks into a rich prompt section."""
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

    Args:
        task_tools: Optional custom task tools.
        formatter: Optional custom function to format tasks into a prompt section.
    """

    def __init__(
        self,
        task_tools: list[BaseTool] | None = None,
        formatter: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        if task_tools is not None:
            self._tools = tuple(task_tools)
        else:
            from langchain_agentkit.extensions.tasks.tools import create_task_tools

            self._tools = tuple(create_task_tools())
        self._formatter = formatter or format_task_context

    @property
    def state_schema(self) -> type:
        from langchain_agentkit.extensions.tasks.state import TasksState

        return TasksState

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        sections = [BASE_AGENT_PROMPT]
        tasks = state.get("tasks") or []
        sections.append(self._formatter(tasks))
        return "\n\n".join(sections)
