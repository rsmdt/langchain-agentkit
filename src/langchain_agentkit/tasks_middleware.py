"""TasksMiddleware — task management tools and system prompt guidance.

Usage::

    from langchain_agentkit import TasksMiddleware

    mw = TasksMiddleware(task_tools=[task_create, task_update])
    mw.tools   # [task_create, task_update]
    mw.prompt(state, config)  # Base agent prompt + task context
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool


BASE_AGENT_PROMPT = """\
## Core Behavior
- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble.
- Focus on solving the user's actual problem, not demonstrating your capabilities.

## Doing Tasks
1. Understand first -- read relevant context, check existing patterns.
2. Act -- implement the solution. Work quickly but accurately.
3. Verify -- check your work against what was asked.

## Progress Updates
For longer tasks, provide brief progress updates at reasonable intervals."""

TASK_MANAGEMENT_PROMPT = """\
## Task Management

You have access to task management tools to help you manage and plan complex objectives.
Use these tools for complex objectives to ensure that you are tracking each necessary step
and giving the user visibility into your progress.

These tools are very helpful for planning complex objectives, and for breaking down larger
complex objectives into smaller steps.

It is critical that you mark tasks as completed as soon as you are done with a step.
Do not batch up multiple steps before marking them as completed.

For simple objectives that only require a few steps, it is better to just complete the
objective directly and NOT use these tools.

## Important Task Usage Notes
- When starting work on a task, set status to `in_progress` BEFORE beginning.
- When finishing, set `completed` ONLY when the work is fully done and verified.
- Don't be afraid to revise the task list as you go. New information may reveal new tasks
  that need to be done, or old tasks that are irrelevant.
- Never mark `completed` if work is partial or errors are unresolved."""


def format_task_context(tasks: list[dict]) -> str:
    """Format active tasks into a prompt section.

    Args:
        tasks: List of task dicts with at least 'subject' and 'status' keys.
    """
    if not tasks:
        return ""
    lines = ["## Current Tasks", ""]
    for task in tasks:
        status = task.get("status", "pending")
        subject = task.get("subject", "Untitled")
        marker = "x" if status == "completed" else " "
        lines.append(f"- [{marker}] {subject} ({status})")
    return "\n".join(lines)


class TasksMiddleware:
    """Middleware providing task management tools and system prompt guidance.

    Tools: Accepts task tools via constructor (injectable dependency).
    Prompt: Base agent behavior + task context or task guidance.

    Example::

        mw = TasksMiddleware(task_tools=[task_create, task_update, task_list])
        mw.tools   # [task_create, task_update, task_list]
        mw.prompt(state, config)  # Task management prompt
    """

    def __init__(self, task_tools: list[BaseTool] | None = None) -> None:
        self._tools = list(task_tools) if task_tools else []

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    def prompt(self, state: dict, config: RunnableConfig) -> str:
        sections = [BASE_AGENT_PROMPT]
        tasks = state.get("tasks") or []
        if tasks:
            sections.append(format_task_context(tasks))
        else:
            sections.append(TASK_MANAGEMENT_PROMPT)
        return "\n\n".join(sections)
