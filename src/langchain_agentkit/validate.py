"""Validation utilities for langchain-agentkit.

Validates skill and agent names against the dasherized-name convention:
1-64 chars, lowercase alphanumeric + hyphens, no leading/trailing/consecutive hyphens.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_agentkit.types import SkillConfig

# Dasherized name: 1-64 chars, lowercase + digits + hyphens, no leading/trailing/consecutive hyphens
NAME_PATTERN = re.compile(r"^[a-z](?:[a-z0-9]|-(?!-)){0,62}[a-z0-9]$|^[a-z]$")

# Backward-compatible alias
SKILL_NAME_PATTERN = NAME_PATTERN


def validate_name(name: str) -> str | None:
    """Validate a dasherized name. Returns error message or None if valid."""
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
    """Validate a skill configuration.

    Returns a list of error messages. An empty list means valid.
    """
    errors: list[str] = []

    name_error = validate_name(config.name)
    if name_error:
        errors.append(f"Skill: {name_error}")

    if not config.description:
        errors.append(f"Skill '{config.name}' is missing required field 'description'")

    return errors
