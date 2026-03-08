"""Validation utilities for langchain-agentkit.

Validates skill configurations against the AgentSkills.io specification.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_agentkit.types import SkillConfig

# AgentSkills.io name constraints:
# 1-64 chars, lowercase alphanumeric + hyphens, no leading/trailing/consecutive hyphens
_NAME_PATTERN = re.compile(r"^[a-z](?:[a-z0-9]|-(?!-)){0,62}[a-z0-9]$|^[a-z]$")


def validate_skill_config(config: SkillConfig) -> list[str]:
    """Validate a skill configuration against the AgentSkills.io spec.

    Returns a list of error messages. An empty list means valid.
    """
    errors: list[str] = []

    if not config.name:
        errors.append("Skill is missing required field 'name'")
    elif not _NAME_PATTERN.match(config.name):
        errors.append(
            f"Skill name '{config.name}' is invalid. "
            f"Must be 1-64 lowercase alphanumeric characters and hyphens, "
            f"no leading/trailing/consecutive hyphens."
        )

    if not config.description:
        errors.append(f"Skill '{config.name}' is missing required field 'description'")

    # AgentSkills.io: name must match parent directory name
    dir_name = config.directory.name
    if config.name and dir_name and config.name != dir_name:
        errors.append(
            f"Skill name '{config.name}' does not match directory name '{dir_name}'. "
            f"AgentSkills.io requires these to match."
        )

    return errors
