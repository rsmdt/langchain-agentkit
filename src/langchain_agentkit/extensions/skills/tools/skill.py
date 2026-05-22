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


_SKILL_TOOL_DESCRIPTION = """Execute a skill in the main conversation. Skills provide specialized capabilities; a user's "slash command" / "/<name>" refers to one.
- Invoke by skill name with optional args (e.g. "summarize" args "--short"; or fully-qualified "ms-office-suite:pdf"). Available skills are listed in the system prompt.
- When a skill matches the request, call it BEFORE any other response — never just mention a skill without calling it.
- Don't invoke an already-running skill; don't use this for built-in CLI commands (/help, /clear). If a <command-name> tag is already in this turn, the skill is loaded — follow its instructions instead of calling again."""


def build_skill_tool(configs: list[SkillConfig]) -> BaseTool:
    """Build the Skill tool from a list of SkillConfig objects.

    The tool description is static. :class:`SkillsExtension` lists the
    available skills in the system prompt via ``_format_skills_list``.
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
