"""Teams extension — message-driven team coordination."""

from langchain_agentkit.extensions.teams.bus import (
    SHUTDOWN_SIGNAL,
    ActiveTeam,
    TeamMessage,
    TeamMessageBus,
    _teammate_loop,
)
from langchain_agentkit.extensions.teams.extension import TeamExtension
from langchain_agentkit.extensions.teams.state import TeamState
from langchain_agentkit.extensions.teams.tools import create_team_tools

__all__ = [
    "ActiveTeam",
    "SHUTDOWN_SIGNAL",
    "TeamExtension",
    "TeamMessage",
    "TeamMessageBus",
    "TeamState",
    "_teammate_loop",
    "create_team_tools",
]
