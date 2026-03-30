"""Skill tool builder for LangGraph agents.

Provides the ``Skill`` tool for progressive disclosure of skill instructions.
The tool is built from a list of ``SkillConfig`` objects by ``build_skill_tool()``.

Usage::

    from langchain_agentkit.tools.skill import build_skill_tool
    from langchain_agentkit.types import SkillConfig

    configs = [SkillConfig(name="research", description="...", prompt="...")]
    tool = build_skill_tool(configs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
)
from pydantic import BaseModel, Field

from langchain_agentkit.validate import NAME_PATTERN

if TYPE_CHECKING:
    from langchain_agentkit.types import SkillConfig


class SkillInput(BaseModel):
    """Input schema for the Skill tool."""

    skill_name: str = Field(description="Name of the skill to load (e.g. 'market-sizing')")


def _build_available_skills_description(configs: list[SkillConfig]) -> str:
    """Build ``<available_skills>`` XML block from skill configs."""
    if not configs:
        return ""

    entries: list[str] = []
    for config in sorted(configs, key=lambda c: c.name):
        entry = (
            f"<skill>\n"
            f"  <name>{config.name}</name>\n"
            f"  <description>{config.description}</description>\n"
            f"</skill>"
        )
        entries.append(entry)

    return "\n\n<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"


def build_skill_tool(configs: list[SkillConfig]) -> BaseTool:
    """Build the Skill tool from a list of SkillConfig objects.

    The tool returns skill instructions as a plain string when invoked
    with a skill name.

    Args:
        configs: List of skill configurations to make available.

    Returns:
        A StructuredTool named ``Skill``.
    """
    index: dict[str, SkillConfig] = {c.name: c for c in configs}

    base_description = (
        "Load a skill's instructions to gain domain expertise. "
        "Call this when you need specialized methodology or procedures."
    )
    available_skills_xml = _build_available_skills_description(configs)
    description = base_description + available_skills_xml

    def skill(skill_name: str) -> str:
        """Load a skill's instructions."""
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
