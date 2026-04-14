"""Skill tool builder for LangGraph agents.

Provides the ``Skill`` tool for progressive disclosure of skill instructions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_agentkit.extensions.skills.types import SkillConfig

# Dasherized name: 1-64 chars, lowercase + digits + hyphens
NAME_PATTERN = re.compile(r"^[a-z](?:[a-z0-9]|-(?!-)){0,62}[a-z0-9]$|^[a-z]$")


class SkillInput(BaseModel):
    """Input schema for the Skill tool."""

    skill_name: str = Field(description="Name of the skill to load (e.g. 'market-sizing')")


_SKILL_TOOL_DESCRIPTION = """Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>" (e.g., "/summarize", "/translate"), they are referring to a skill. Use this tool to invoke it.

How to invoke:
- Use this tool with the skill name and optional arguments
- Examples:
  - `skill: "pdf"` - invoke the pdf skill
  - `skill: "summarize", args: "--short"` - invoke with arguments
  - `skill: "translate", args: "fr"` - invoke with arguments
  - `skill: "ms-office-suite:pdf"` - invoke using fully qualified name

Important:
- Available skills are listed in system-reminder messages in the conversation
- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again"""


def _build_available_skills_description(
    configs: list[SkillConfig],
    *,
    max_description_chars: int | None = None,
) -> str:
    """Build available skills list from skill configs."""
    if not configs:
        return ""
    entries: list[str] = []
    for config in sorted(configs, key=lambda c: c.name):
        desc = config.description
        if max_description_chars is not None and len(desc) > max_description_chars:
            desc = desc[: max_description_chars - 1] + "…"
        entries.append(f"- {config.name}: {desc}")
    return "\n\nAvailable skills:\n" + "\n".join(entries)


def render_skills_roster(
    configs: list[SkillConfig],
    *,
    budget_percent: float | None = None,
    max_description_chars: int | None = None,
    context_window: int | None = None,
) -> str:
    """Render the available-skills roster block for the reminder channel.

    Returns an empty string when no skills are configured. Applies the same
    budget/truncation rules that previously ran at tool-build time.
    """
    roster = _build_available_skills_description(
        configs,
        max_description_chars=max_description_chars,
    )
    if not roster:
        return ""
    if budget_percent is not None and context_window is not None:
        budget_chars = int(context_window * budget_percent)
        if len(roster) > budget_chars:
            roster = roster[:budget_chars]
    return roster


def build_skill_tool(
    configs: list[SkillConfig],
    *,
    budget_percent: float | None = None,  # noqa: ARG001 — reminder-side now
    max_description_chars: int | None = None,  # noqa: ARG001
    context_window: int | None = None,  # noqa: ARG001
) -> BaseTool:
    """Build the Skill tool from a list of SkillConfig objects.

    The tool description is static; the available-skills roster is delivered
    separately via :func:`render_skills_roster` (emitted by
    :class:`SkillsExtension` as a ``"reminder"`` dict entry from ``prompt()``).
    """
    index: dict[str, SkillConfig] = {c.name: c for c in configs}

    description = _SKILL_TOOL_DESCRIPTION

    def skill(skill_name: str) -> str:
        if not NAME_PATTERN.match(skill_name):
            available = sorted(index.keys())
            raise ToolException(
                f"Invalid skill name '{skill_name}'. "
                f"Available skills: {', '.join(available) or 'none'}"
            )
        config = index.get(skill_name)
        if config is None:
            available = sorted(index.keys())
            raise ToolException(
                f"Skill '{skill_name}' not found. "
                f"Available skills: {', '.join(available) or 'none'}"
            )
        return config.prompt

    return StructuredTool.from_function(
        func=skill,
        name="Skill",
        description=description,
        args_schema=SkillInput,
        handle_tool_error=True,
    )
