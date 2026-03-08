"""SkillsMiddleware — wraps SkillKit with system prompt injection.

Usage::

    from langchain_agentkit import SkillsMiddleware

    mw = SkillsMiddleware("skills/")
    tools = mw.tools           # [Skill, SkillRead]
    prompt = mw.prompt(state, config)  # Skills system prompt with skill list
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_agentkit.skill_kit import SkillKit
from langchain_agentkit.types import SkillConfig

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool


# fmt: off
SKILLS_SYSTEM_PROMPT = """\
## Skills System

You have access to a skills library that provides specialized methodology and domain knowledge.

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern -- you see their name and \
description above, but only load full instructions when needed:

1. **Recognize when a skill applies**: Check if the current task matches a skill's description
2. **Load the skill**: Call `Skill("skill-name")` to get full instructions
3. **Follow the skill's instructions**: Contains step-by-step workflows, quality rubrics, \
and methodology
4. **Access supporting files**: Call `SkillRead("skill-name", "reference/file.md")` for \
templates, examples, or rubrics

**When to Use Skills:**
- Starting work on an artifact that has a matching skill
- You need structured methodology or quality criteria
- A skill provides proven patterns for the task at hand

**When NOT to Use Skills:**
- Conversational exchanges or simple questions
- You already loaded the skill this session (don't reload)
- The task doesn't match any available skill"""
# fmt: on


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
        self._kit = SkillKit(skills_dirs)

    @property
    def tools(self) -> list[BaseTool]:
        return self._kit.tools

    def prompt(self, state: dict, config: RunnableConfig) -> str:
        return SKILLS_SYSTEM_PROMPT.format(
            skills_list=self._format_skills_list(),
        )

    def _format_skills_list(self) -> str:
        skills = self._kit._build_skill_index()
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
