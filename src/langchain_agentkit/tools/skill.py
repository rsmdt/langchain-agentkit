"""Backward-compat shim. Import from langchain_agentkit.extensions.skills.tools instead."""

from langchain_agentkit.extensions.skills.tools import *  # noqa: F401, F403
from langchain_agentkit.extensions.skills.tools import build_skill_tool as build_skill_tool
