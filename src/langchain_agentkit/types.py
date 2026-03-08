"""Configuration types for skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from langchain_agentkit.frontmatter import parse_frontmatter


@dataclass(frozen=True)
class SkillConfig:
    """Parsed configuration for an agent skill.

    Frontmatter fields:
        name: Skill identifier (lowercase, hyphens).
        description: One-line summary shown in the Skill tool's available_skills list.

    The body content is returned as instructions when the skill is loaded.
    """

    name: str
    description: str
    instructions: str = ""
    directory: Path = field(default=Path("."))
    reference_files: list[str] = field(default_factory=list)

    @classmethod
    def from_directory(cls, skill_dir: Path) -> SkillConfig:
        """Parse a skill from its directory containing SKILL.md.

        Raises:
            FileNotFoundError: If ``SKILL.md`` does not exist in *skill_dir*.
        """
        skill_md = Path(skill_dir) / "SKILL.md"
        result = parse_frontmatter(skill_md)
        metadata = result.metadata

        ref_files = [
            f.name for f in Path(skill_dir).iterdir() if f.is_file() and f.name != "SKILL.md"
        ]

        return cls(
            name=metadata.get("name", skill_dir.name),
            description=metadata.get("description", ""),
            instructions=result.content,
            directory=Path(skill_dir),
            reference_files=sorted(ref_files),
        )
