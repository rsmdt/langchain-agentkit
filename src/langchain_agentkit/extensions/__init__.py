"""Extension implementations for langchain-agentkit.

Each extension is a self-contained package under extensions/:

    from langchain_agentkit.extensions.skills import SkillsExtension
    from langchain_agentkit.extensions.agents import AgentExtension
    from langchain_agentkit.extensions.tasks import TasksExtension
    from langchain_agentkit.extensions.teams import TeamExtension
    from langchain_agentkit.extensions.filesystem import FilesystemExtension
    from langchain_agentkit.extensions.hitl import HITLExtension
    from langchain_agentkit.extensions.web_search import WebSearchExtension
"""

from langchain_agentkit.extensions.agents import AgentExtension as AgentExtension
from langchain_agentkit.extensions.filesystem import (
    FilesystemExtension as FilesystemExtension,
)
from langchain_agentkit.extensions.hitl import HITLExtension as HITLExtension
from langchain_agentkit.extensions.skills import SkillsExtension as SkillsExtension
from langchain_agentkit.extensions.tasks import TasksExtension as TasksExtension
from langchain_agentkit.extensions.teams import TeamExtension as TeamExtension
from langchain_agentkit.extensions.web_search import (
    DuckDuckGoSearchProvider as DuckDuckGoSearchProvider,
    QwantSearchProvider as QwantSearchProvider,
    WebSearchExtension as WebSearchExtension,
)

__all__ = [
    "AgentExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HITLExtension",
    "QwantSearchProvider",
    "SkillsExtension",
    "TasksExtension",
    "TeamExtension",
    "WebSearchExtension",
]
