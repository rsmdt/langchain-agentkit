"""Skill discovery from filesystem directories and backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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


def _strip_line_numbers(formatted: str) -> str:
    """Strip line-number prefixes from BackendProtocol.read() output."""
    lines = []
    for line in formatted.splitlines(keepends=True):
        _, _, content = line.partition("\t")
        lines.append(content)
    return "".join(lines)


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    """Parse a markdown file with YAML frontmatter. Returns (metadata, content)."""
    from langchain_agentkit.frontmatter import parse_frontmatter

    result = parse_frontmatter(path)
    return result.metadata, result.content


def _parse_frontmatter_string(text: str) -> tuple[dict[str, Any], str]:
    """Parse a string with YAML frontmatter. Returns (metadata, content)."""
    from langchain_agentkit.frontmatter import parse_frontmatter_string

    result = parse_frontmatter_string(text)
    return result.metadata, result.content


def discover_skills_from_directory(path: Path) -> list[SkillConfig]:
    """Discover skills by scanning a local directory for SKILL.md files."""
    if not path.is_dir():
        return []
    configs: list[SkillConfig] = []
    seen_names: set[str] = set()
    for skill_file in sorted(path.rglob("SKILL.md")):
        try:
            metadata, content = _parse_frontmatter(skill_file)
        except (OSError, UnicodeDecodeError):
            logger.warning("Skipping unreadable skill file: %s", skill_file)
            continue
        if not metadata:
            logger.warning("Skipping skill without frontmatter: %s", skill_file)
            continue
        config = SkillConfig.from_frontmatter(metadata, content)
        errors = validate_skill_config(config)
        if errors:
            logger.warning("Skipping invalid skill %s: %s", skill_file, "; ".join(errors))
            continue
        if config.name in seen_names:
            logger.warning("Skipping duplicate skill name '%s': %s", config.name, skill_file)
            continue
        seen_names.add(config.name)
        configs.append(config)
    return configs


def discover_skills_from_backend(backend: BackendProtocol, path: str) -> list[SkillConfig]:
    """Discover skills via a BackendProtocol by globbing for SKILL.md files."""
    matches = backend.glob("**/SKILL.md", path=path)
    configs: list[SkillConfig] = []
    seen_names: set[str] = set()
    for match in sorted(matches):
        try:
            formatted = backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            logger.warning("Skipping unreadable skill file: %s", match)
            continue
        content = _strip_line_numbers(formatted)
        metadata, body = _parse_frontmatter_string(content)
        if not metadata:
            logger.warning("Skipping skill without frontmatter: %s", match)
            continue
        config = SkillConfig.from_frontmatter(metadata, body)
        errors = validate_skill_config(config)
        if errors:
            logger.warning("Skipping invalid skill %s: %s", match, "; ".join(errors))
            continue
        if config.name in seen_names:
            logger.warning("Skipping duplicate skill name '%s': %s", config.name, match)
            continue
        seen_names.add(config.name)
        configs.append(config)
    return configs
