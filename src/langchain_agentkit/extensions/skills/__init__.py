"""Skills extension — progressive disclosure of skill instructions."""

from langchain_agentkit.extensions.skills.discovery import (
    validate_name,
    validate_skill_config,
)
from langchain_agentkit.extensions.skills.extension import SkillsExtension
from langchain_agentkit.extensions.skills.tools import build_skill_tool
from langchain_agentkit.extensions.skills.types import SkillConfig

__all__ = [
    "SkillConfig",
    "SkillsExtension",
    "build_skill_tool",
    "validate_name",
    "validate_skill_config",
]
