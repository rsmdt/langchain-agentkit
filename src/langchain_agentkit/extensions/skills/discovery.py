"""Skill discovery from filesystem directories and backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extensions.discovery import (
    discover_from_backend,
    discover_from_directory,
)
from langchain_agentkit.extensions.skills.types import SkillConfig

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_agentkit.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


def validate_name(name: str) -> str | None:
    """Validate a dasherized name. Returns error message or None if valid."""
    from langchain_agentkit.extensions.skills.tools import NAME_PATTERN

    if not name:
        return "Name is required"
    if not NAME_PATTERN.match(name):
        return (
            f"Name '{name}' is invalid. "
            f"Must be 1-64 lowercase alphanumeric characters and hyphens, "
            f"no leading/trailing/consecutive hyphens."
        )
    return None


def validate_skill_config(config: SkillConfig) -> list[str]:
    """Validate a skill configuration. Returns list of error messages."""
    errors: list[str] = []
    name_error = validate_name(config.name)
    if name_error:
        errors.append(f"Skill: {name_error}")
    if not config.description:
        errors.append(f"Skill '{config.name}' is missing required field 'description'")
    return errors


def _parse_skill(metadata: dict[str, Any], content: str) -> SkillConfig | None:
    """Parse and validate a SkillConfig from frontmatter."""
    config = SkillConfig.from_frontmatter(metadata, content)
    errors = validate_skill_config(config)
    if errors:
        logger.warning("Invalid skill '%s': %s", config.name, "; ".join(errors))
        return None
    return config


def discover_skills_from_directory(path: Path) -> list[SkillConfig]:
    """Discover skills by scanning a local directory for SKILL.md files."""
    return discover_from_directory(
        path,
        file_pattern="SKILL.md",
        parser=_parse_skill,
        namer=lambda c: c.name,
        label="skill",
    )


async def discover_skills_from_backend(backend: BackendProtocol, path: str) -> list[SkillConfig]:
    """Discover skills via a BackendProtocol by globbing for SKILL.md files."""
    return await discover_from_backend(
        backend,
        path,
        file_pattern="**/SKILL.md",
        parser=_parse_skill,
        namer=lambda c: c.name,
        label="skill",
    )
