"""SkillsMiddleware — skill loading with optional filesystem integration.

Provides the ``Skill`` tool for progressive disclosure of skill instructions.
Optionally includes filesystem tools (Read, Write, Edit, Glob, Grep) when
no external VFS is provided.

Usage::

    # Convenience: auto-includes filesystem tools
    mw = SkillsMiddleware(skills="skills/")
    mw.tools   # [Skill, Read, Write, Edit, Glob, Grep]

    # Explicit: provide shared VFS, manage filesystem tools separately
    vfs = VirtualFilesystem()
    mw = SkillsMiddleware(skills="skills/", filesystem=vfs)
    mw.tools   # [Skill] only
    fs = FilesystemMiddleware(vfs)  # provides Read, Write, Edit, Glob, Grep
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.tools.filesystem import create_filesystem_tools
from langchain_agentkit.tools.skill import SkillRegistry
from langchain_agentkit.types import SkillConfig
from langchain_agentkit.vfs import VirtualFilesystem

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_skills_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "skills_system.md")


class SkillsMiddleware:
    """Middleware providing skill tools and optional filesystem access.

    Loads skills from real filesystem directories into a
    :class:`VirtualFilesystem`. The ``Skill`` tool is always provided.
    Filesystem tools are included only when no external VFS is given.

    Args:
        skills: Path(s) to directories containing skill subdirectories.
        filesystem: Optional shared VirtualFilesystem. When provided,
            skills are populated into it but filesystem tools are NOT
            included — the caller manages file tools via
            ``FilesystemMiddleware(filesystem)``. When ``None``, an
            internal VFS is created and filesystem tools are bundled.
        skills_base_path: Virtual path prefix for skills. Default ``/skills``.
    """

    def __init__(
        self,
        skills: str | Path | list[str | Path],
        filesystem: VirtualFilesystem | None = None,
        skills_base_path: str = "/skills",
    ) -> None:
        self._registry = SkillRegistry(skills)
        self._skills_base_path = skills_base_path

        if filesystem is not None:
            self._filesystem = filesystem
            self._owns_filesystem = False
        else:
            self._filesystem = VirtualFilesystem()
            self._owns_filesystem = True

        self._registry.populate_filesystem(
            self._filesystem,
            base_path=skills_base_path,
        )
        self._tools_cache: list[BaseTool] | None = None

    @property
    def filesystem(self) -> VirtualFilesystem:
        """The VirtualFilesystem containing skill files."""
        return self._filesystem

    @property
    def state_schema(self) -> None:
        """No additional state keys — skills are in-memory."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """Skill tool, plus filesystem tools if VFS is internally owned."""
        if self._tools_cache is None:
            tools = list(self._registry.tools)
            if self._owns_filesystem:
                tools.extend(create_filesystem_tools(self._filesystem))
            self._tools_cache = tools
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
