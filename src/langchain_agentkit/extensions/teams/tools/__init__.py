"""Teams tools package.

Command-based team coordination tools for LangGraph agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_agentkit.extensions.teams.tools.team_create import build_team_create_tool
from langchain_agentkit.extensions.teams.tools.team_dissolve import build_team_dissolve_tool
from langchain_agentkit.extensions.teams.tools.team_message import build_team_message_tool
from langchain_agentkit.extensions.teams.tools.team_status import build_team_status_tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.extension import TeamExtension


def create_team_tools(ext: TeamExtension) -> list[BaseTool]:
    """Create Command-based team coordination tools.

    Returns four tools: TeamCreate, TeamMessage, TeamStatus, TeamDissolve.
    """
    return [
        build_team_create_tool(ext),
        build_team_message_tool(ext),
        build_team_status_tool(ext),
        build_team_dissolve_tool(ext),
    ]


__all__ = ["create_team_tools"]
