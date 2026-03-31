"""Backward-compat shim. Import from langchain_agentkit.extensions.skills instead."""

from langchain_agentkit.extensions.skills.discovery import (
    validate_name as validate_name,
    validate_skill_config as validate_skill_config,
)
from langchain_agentkit.extensions.skills.tools import NAME_PATTERN as NAME_PATTERN
