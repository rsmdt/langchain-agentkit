"""SkillsMiddleware — skills + virtual filesystem for LangGraph agents.

Loads skills from real filesystem into a virtual filesystem, then provides:
- ``Skill`` tool for loading skill instructions (progressive disclosure)
- ``Read``, ``Write``, ``Edit``, ``Glob``, ``Grep`` tools on the virtual filesystem

Usage::

    from langchain_agentkit import SkillsMiddleware

    mw = SkillsMiddleware("skills/")
    mw.tools   # [Skill, Read, Write, Edit, Glob, Grep]
    mw.prompt(state, runtime)  # Skills system prompt with skill list

Reference files are accessible via ``Read("/skills/market-sizing/calculator.py")``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.filesystem_tools import create_filesystem_tools
from langchain_agentkit.skill_registry import SkillRegistry
from langchain_agentkit.types import SkillConfig
from langchain_agentkit.virtual_filesystem import VirtualFilesystem

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_skills_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "skills_system.md")


class SkillsMiddleware:
    """Middleware providing skill tools and virtual filesystem access.

    Loads skills from real filesystem directories into an in-memory
    :class:`VirtualFilesystem` at ``/skills/{name}/``. Provides:

    - **Skill**: Load skill instructions (progressive disclosure shortcut)
    - **Read, Write, Edit, Glob, Grep**: Virtual filesystem tools

    The ``SkillRead`` tool is replaced by ``Read`` — reference files are
    accessible at ``/skills/{skill_name}/{filename}``.

    Args:
        skills_dirs: Path(s) to directories containing skill subdirectories.
        filesystem: Optional pre-configured VirtualFilesystem. If ``None``,
            creates a new one and populates it with skill files.
        skills_base_path: Virtual path prefix for skills. Default ``/skills``.

    Example::

        mw = SkillsMiddleware("skills/")
        mw.tools   # [Skill, Read, Write, Edit, Glob, Grep]
    """

    def __init__(
        self,
        skills_dirs: str | Path | list[str | Path],
        filesystem: VirtualFilesystem | None = None,
        skills_base_path: str = "/skills",
    ) -> None:
        self._registry = SkillRegistry(skills_dirs)
        self._skills_base_path = skills_base_path
        self.filesystem = filesystem or VirtualFilesystem()
        self._registry.populate_filesystem(self.filesystem, base_path=skills_base_path)
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """Tools: ``[Skill, Read, Write, Edit, Glob, Grep]``."""
        if self._tools_cache is None:
            skill_tools = self._registry.tools
            fs_tools = create_filesystem_tools(self.filesystem)
            self._tools_cache = skill_tools + fs_tools
        return self._tools_cache

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Skills system prompt with formatted skills list."""
        return _skills_system_prompt.format(
            skills_list=self._format_skills_list(),
        )

    def _format_skills_list(self) -> str:
        skills = self._registry.skill_index
        if not skills:
            return "(No skills available)"

        lines = []
        for _name, skill_dir in sorted(skills.items()):
            config = SkillConfig.from_directory(skill_dir)
            lines.append(f"- **{config.name}**: {config.description}")
            vfs_path = f"{self._skills_base_path}/{config.name}"
            lines.append(f'  -> Load via Skill("{config.name}")')
            if config.reference_files:
                files_str = ", ".join(config.reference_files)
                lines.append(f"  -> Reference files: {files_str}")
                lines.append(f'  -> Read via Read("{vfs_path}/{{filename}}")')
        return "\n".join(lines)
