"""TasksExtension — task management tools and system prompt guidance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent

_task_management_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "prompt.md")

TASK_MANAGEMENT_PROMPT = _task_management_prompt.format()


# Returns a focus directive naming the in-progress task(s), or "" when none.
# The full task list is deliberately not repeated — it is already in the
# conversation via the task-tool calls, and completed/old tasks are noise.
def format_task_context(tasks: list[dict[str, Any]]) -> str:
    in_progress = [t.get("subject", "") for t in tasks if t.get("status") == "in_progress"]
    if not in_progress:
        return ""
    listed = "\n".join(f"- {subject}" for subject in in_progress)
    return f"Keep your focus on:\n{listed}"


class TasksExtension(Extension):
    """Extension providing task management tools and system prompt guidance.

    Args:
        tools: Optional explicit tool list. When provided, replaces the
            default task-management tool set entirely; the extension will
            not rebuild them in ``setup()`` for team-awareness. When
            ``None``, the extension builds its defaults (TaskCreate,
            TaskUpdate, TaskList, TaskGet, TaskStop) and upgrades their
            descriptions if a ``TeamExtension`` sibling is detected.
        formatter: Optional custom function that renders the per-turn reminder
            from the current tasks. It must return ``""`` when there is nothing
            to surface (the default grounds on the in-progress task). The
            static task-management guidance goes to the system prompt
            separately.
    """

    def __init__(
        self,
        *,
        tools: Sequence[BaseTool] | None = None,
        formatter: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        self._custom_tools: tuple[BaseTool, ...] | None = (
            tuple(tools) if tools is not None else None
        )
        self._tools: tuple[BaseTool, ...] = ()
        self._formatter = formatter or format_task_context
        # Build default tools; setup() may rebuild with team-aware descriptions.
        self._build_tools(team_active=False)

    def _build_tools(self, *, team_active: bool) -> None:
        if self._custom_tools is not None:
            self._tools = self._custom_tools
            return
        from langchain_agentkit.extensions.tasks.tools import create_task_tools

        self._tools = tuple(create_task_tools(team_active=team_active))

    @override
    def setup(  # type: ignore[override]
        self, *, extensions: list[Extension], **_: Any
    ) -> None:
        from langchain_agentkit.extensions.teams import TeamExtension

        has_team = any(isinstance(e, TeamExtension) for e in extensions)
        self._build_tools(team_active=has_team)

    @property
    @override
    def state_schema(self) -> type:
        from langchain_agentkit.extensions.tasks.state import TasksState

        return TasksState

    @property
    @override
    def tools(self) -> list[BaseTool]:
        return self._tools  # type: ignore[return-value]

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str | dict[str, str]:
        # Static guidance always rides the cacheable system prompt; the live
        # task list (per-turn dynamic) goes to the reminder when non-empty.
        rendered = self._formatter(state.get("tasks") or [])
        if not rendered:
            return TASK_MANAGEMENT_PROMPT
        return {"prompt": TASK_MANAGEMENT_PROMPT, "reminder": rendered}
