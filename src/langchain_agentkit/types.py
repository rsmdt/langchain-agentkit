"""Configuration types for skills and agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillConfig:
    """Parsed configuration for a skill.

    Frontmatter fields:
        name: Skill identifier (lowercase, hyphens, AgentSkills.io compliant).
        description: One-line summary shown in the Skill tool's available_skills list.

    The body content is returned as instructions when the skill is loaded.
    """

    name: str
    description: str
    instructions: str = ""

    @classmethod
    def from_frontmatter(cls, metadata: dict[str, str], content: str) -> SkillConfig:
        """Create a SkillConfig from parsed frontmatter metadata and body content.

        Args:
            metadata: Dict from YAML frontmatter (expects ``name``, ``description``).
            content: Markdown body content (becomes instructions).
        """
        return cls(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            instructions=content,
        )
