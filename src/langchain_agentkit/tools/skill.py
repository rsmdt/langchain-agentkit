"""SkillRegistry — provides Skill tool and filesystem population for LangGraph agents.

Usage::

    from langchain_agentkit import SkillRegistry

    # Single directory
    registry = SkillRegistry("skills/")

    # Multiple directories
    registry = SkillRegistry(["skills/", "shared_skills/"])

    # Get tools for manual LangGraph wiring
    tools = registry.tools  # → [Skill]

    # Populate a virtual filesystem with skill files
    from langchain_agentkit.vfs import VirtualFilesystem
    vfs = VirtualFilesystem()
    registry.populate_filesystem(vfs)

The ``Skill`` tool returns skill instructions as a plain string.
Reference files are accessible via the virtual filesystem ``Read`` tool.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
)
from pydantic import BaseModel, Field

from langchain_agentkit.types import SkillConfig

if TYPE_CHECKING:
    from langchain_agentkit.vfs import VirtualFilesystem

# AgentSkills.io: 1-64 chars, lowercase + digits + hyphens, no leading/trailing/consecutive hyphens
SKILL_NAME_PATTERN = re.compile(r"^[a-z](?:[a-z0-9]|-(?!-)){0,62}[a-z0-9]$|^[a-z]$")
class SkillInput(BaseModel):
    """Input schema for the Skill tool."""

    skill_name: str = Field(description="Name of the skill to load (e.g. 'market-sizing')")


class SkillRegistry:
    """Registry providing ``Skill`` and ``SkillRead`` tools.

    Scans one or more directories for skill subdirectories containing
    ``SKILL.md`` files. Provides two tools via the ``tools`` property:

    - **Skill**: Loads a skill's instructions. The tool description
      dynamically lists all available skills for semantic discovery.
    - **SkillRead**: Reads a reference file from within a skill's directory.

    Example::

        from langchain_agentkit import SkillRegistry

        registry = SkillRegistry("skills/")

        # Use in any LangGraph setup
        all_tools = [web_search, calculate] + registry.tools
        bound_llm = llm.bind_tools(all_tools)

    Args:
        skills_dirs: A single directory path or list of directory paths
            containing skill subdirectories.
    """

    def __init__(self, skills_dirs: str | Path | list[str | Path]) -> None:
        """Create a SkillRegistry from one or more skill directories.

        Args:
            skills_dirs: A single path (str or Path) or list of paths to
                directories containing skill subdirectories with ``SKILL.md``
                files.
        """
        if isinstance(skills_dirs, (str, Path)):
            self.skills_dirs: list[str | Path] = [skills_dirs]
        else:
            self.skills_dirs = list(skills_dirs)
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """The registry's tools: ``[Skill]``.

        Built once on first access, then cached.
        """
        if self._tools_cache is None:
            self._tools_cache = [self._build_skill_tool()]
        return self._tools_cache

    @property
    def skill_index(self) -> dict[str, Path]:
        """Mapping from frontmatter skill name to skill directory path."""
        return self._build_skill_index()

    def _resolve_skills_dirs(self) -> list[Path]:
        return [Path(d).resolve() for d in self.skills_dirs if d]

    def _validate_skill_name(self, skill_name: str) -> None:
        if not SKILL_NAME_PATTERN.match(skill_name):
            available = self._list_skills()
            raise ToolException(
                f"Invalid skill name '{skill_name}'. "
                f"Available skills: {', '.join(available) or 'none'}"
            )

    def _validate_path_traversal(self, resolved: Path, base: Path) -> None:
        if not str(resolved).startswith(str(base) + os.sep):
            raise ToolException("Path traversal detected")

    def _build_skill_index(self) -> dict[str, Path]:
        """Build a mapping from frontmatter skill name → skill directory path.

        Scans all skill directories and reads each SKILL.md frontmatter
        to get the canonical name. First directory wins on name collisions.
        """
        index: dict[str, Path] = {}
        for skills_dir in self._resolve_skills_dirs():
            if not skills_dir.exists():
                continue
            for d in skills_dir.iterdir():
                if d.is_dir() and (d / "SKILL.md").exists():
                    config = SkillConfig.from_directory(d)
                    if config.name not in index:
                        index[config.name] = d
        return index

    def _list_skills(self) -> list[str]:
        return sorted(self._build_skill_index().keys())

    def _find_skill_dir(self, skill_name: str) -> Path | None:
        """Find the skill directory for a given frontmatter skill name."""
        return self._build_skill_index().get(skill_name)

    def _build_available_skills_description(self) -> str:
        """Build ``<available_skills>`` XML block from all skills directories."""
        skill_names = self._list_skills()
        if not skill_names:
            return ""

        entries: list[str] = []
        for name in skill_names:
            skill_dir = self._find_skill_dir(name)
            if skill_dir is None:
                continue
            config = SkillConfig.from_directory(skill_dir)
            entry = (
                f"<skill>\n"
                f"  <name>{config.name}</name>\n"
                f"  <description>{config.description}</description>\n"
            )
            if config.reference_files:
                files_str = ", ".join(config.reference_files)
                entry += f"  <reference_files>{files_str}</reference_files>\n"
            entry += "</skill>"
            entries.append(entry)

        return "\n\n<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"

    def populate_filesystem(
        self, filesystem: VirtualFilesystem, base_path: str = "/skills",
    ) -> None:
        """Load all skill files into a :class:`VirtualFilesystem`.

        Copies each skill's ``SKILL.md`` and reference files from the real
        filesystem into the virtual filesystem at ``{base_path}/{skill_name}/``.

        Args:
            filesystem: Target virtual filesystem to populate.
            base_path: Virtual directory prefix for skills.
        """
        for name, skill_dir in self._build_skill_index().items():
            vfs_dir = f"{base_path}/{name}"

            # Load SKILL.md
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                filesystem.write(f"{vfs_dir}/SKILL.md", skill_md.read_text())

            # Load all reference files (non-SKILL.md files)
            for file_path in sorted(skill_dir.iterdir()):
                if file_path.is_file() and file_path.name != "SKILL.md":
                    try:
                        content = file_path.read_text()
                    except UnicodeDecodeError:
                        content = f"(binary file: {file_path.name})"
                    filesystem.write(f"{vfs_dir}/{file_path.name}", content)

    def _build_skill_tool(self) -> StructuredTool:
        base_description = (
            "Load a skill's instructions to gain domain expertise. "
            "Call this when you need specialized methodology or procedures."
        )
        available_skills_xml = self._build_available_skills_description()
        description = base_description + available_skills_xml

        def skill(skill_name: str) -> str:
            """Load a skill's instructions."""
            self._validate_skill_name(skill_name)

            skill_dir = self._find_skill_dir(skill_name)
            if skill_dir is None:
                available = self._list_skills()
                raise ToolException(
                    f"Skill '{skill_name}' not found. "
                    f"Available skills: {', '.join(available) or 'none'}"
                )

            skill_path = (skill_dir / "SKILL.md").resolve()
            self._validate_path_traversal(skill_path, skill_dir.parent)

            config = SkillConfig.from_directory(skill_dir)
            return config.instructions

        return StructuredTool.from_function(
            func=skill,
            name="Skill",
            description=description,
            args_schema=SkillInput,
            handle_tool_error=True,
        )

