"""SkillsMiddleware — wraps SkillRegistry with system prompt injection.

Usage::

    from langchain_agentkit import SkillsMiddleware

    mw = SkillsMiddleware("skills/")
    tools = mw.tools           # [Skill, SkillRead]
    prompt = mw.prompt(state, runtime)  # Skills system prompt with skill list
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.skill_registry import SkillRegistry
from langchain_agentkit.types import SkillConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.runtime import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_skills_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "skills_system.md")


class SkillsMiddleware:
    """Middleware providing skill tools and system prompt guidance.

    Adapts the Deep Agents SkillsMiddleware pattern:
    - Tools: Skill (load instructions) + SkillRead (read reference files)
    - Prompt: Progressive disclosure guidance + available skills list

    Example::

        mw = SkillsMiddleware("skills/")
        mw.tools   # [Skill, SkillRead]
        mw.prompt(state, config)  # Skills system prompt with skill list
    """

    def __init__(self, skills_dirs: str | list[str]) -> None:
        self._kit = SkillRegistry(skills_dirs)

    @property
    def tools(self) -> list[BaseTool]:
        return self._kit.tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime) -> str:
        return _skills_system_prompt.format(
            skills_list=self._format_skills_list(),
        )

    def _format_skills_list(self) -> str:
        skills = self._kit.skill_index
        if not skills:
            return "(No skills available)"

        lines = []
        for _name, skill_dir in sorted(skills.items()):
            skill_config = SkillConfig.from_directory(skill_dir)
            lines.append(f"- **{skill_config.name}**: {skill_config.description}")
            lines.append(
                f'  -> Load via Skill("{skill_config.name}"), '
                f'read references via SkillRead("{skill_config.name}", "reference/file.md")'
            )
        return "\n".join(lines)
