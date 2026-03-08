"""SkillKit — toolkit providing Skill and SkillRead tools for LangGraph agents.

Usage::

    from langchain_agentkit import SkillKit

    # Single directory
    kit = SkillKit("skills/")

    # Multiple directories
    kit = SkillKit(["skills/", "shared_skills/"])

    # Get tools for manual LangGraph wiring
    tools = kit.tools  # → [Skill, SkillRead]

The ``Skill`` tool returns skill instructions as a plain string.
The ``SkillRead`` tool reads reference files scoped to a skill's directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
)
from pydantic import BaseModel, Field

from langchain_agentkit.types import SkillConfig

# AgentSkills.io: 1-64 chars, lowercase + digits + hyphens, no leading/trailing/consecutive hyphens
SKILL_NAME_PATTERN = re.compile(r"^[a-z](?:[a-z0-9]|-(?!-)){0,62}[a-z0-9]$|^[a-z]$")
REFERENCE_NAME_PATTERN = re.compile(r"^(?!\.\.?$)[a-zA-Z0-9_.-]{1,255}$")


class SkillInput(BaseModel):
    """Input schema for the Skill tool."""

    skill_name: str = Field(description="Name of the skill to load (e.g. 'market-sizing')")


class SkillReadInput(BaseModel):
    """Input schema for the SkillRead tool."""

    skill_name: str = Field(description="Name of the skill containing the reference file")
    file_name: str = Field(description="Name of the reference file to read (e.g. 'calculator.py')")


class SkillKit:
    """Toolkit providing ``Skill`` and ``SkillRead`` tools.

    Scans one or more directories for skill subdirectories containing
    ``SKILL.md`` files. Provides two tools via the ``tools`` property:

    - **Skill**: Loads a skill's instructions. The tool description
      dynamically lists all available skills for semantic discovery.
    - **SkillRead**: Reads a reference file from within a skill's directory.

    Example::

        from langchain_agentkit import SkillKit

        kit = SkillKit("skills/")

        # Use in any LangGraph setup
        all_tools = [web_search, calculate] + kit.tools
        bound_llm = llm.bind_tools(all_tools)

    Args:
        skills_dirs: A single directory path or list of directory paths
            containing skill subdirectories.
    """

    def __init__(self, skills_dirs: str | list[str]) -> None:
        """Create a SkillKit from one or more skill directories.

        Args:
            skills_dirs: A single path or list of paths to directories
                containing skill subdirectories with ``SKILL.md`` files.
        """
        self.skills_dirs = [skills_dirs] if isinstance(skills_dirs, str) else list(skills_dirs)
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """The toolkit's tools: ``[Skill, SkillRead]``.

        Built once on first access, then cached.
        """
        if self._tools_cache is None:
            self._tools_cache = [self._build_skill_tool(), self._build_skill_read_tool()]
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
            entries.append(
                f"<skill>\n"
                f"  <name>{config.name}</name>\n"
                f"  <description>{config.description}</description>\n"
                f"</skill>"
            )

        return "\n\n<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"

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

    def _build_skill_read_tool(self) -> StructuredTool:
        def skill_read(skill_name: str, file_name: str) -> str:
            """Read a reference file from within a skill directory."""
            self._validate_skill_name(skill_name)
            if not REFERENCE_NAME_PATTERN.match(file_name):
                raise ToolException(f"Invalid file name '{file_name}'")

            skill_dir = self._find_skill_dir(skill_name)
            if skill_dir is None:
                raise ToolException(f"Skill '{skill_name}' not found")

            file_path = (skill_dir / file_name).resolve()
            self._validate_path_traversal(file_path, skill_dir.parent)

            if not file_path.exists():
                raise ToolException(
                    f"Reference file '{file_name}' not found in skill '{skill_name}'"
                )

            return file_path.read_text()

        return StructuredTool.from_function(
            func=skill_read,
            name="SkillRead",
            description=(
                "Read a reference file (template, example, script) from within a skill directory."
            ),
            args_schema=SkillReadInput,
            handle_tool_error=True,
        )
