"""Configuration types for skills."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillConfig:
    """Parsed configuration for a skill.

    Frontmatter fields:
        name: Skill identifier (lowercase, hyphens, AgentSkills.io compliant).
        description: One-line summary shown in the Skill tool's available_skills list.

    The body content becomes the prompt returned when the skill is loaded.
    """

    name: str
    description: str
    prompt: str = ""

    @classmethod
    def from_frontmatter(cls, metadata: dict[str, str], content: str) -> SkillConfig:
        """Create a SkillConfig from parsed frontmatter metadata and body content.

        Args:
            metadata: Dict from YAML frontmatter (expects ``name``, ``description``).
            content: Markdown body content (becomes prompt).
        """
        return cls(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            prompt=content,
        )
