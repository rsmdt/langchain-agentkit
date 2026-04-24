"""Extension implementations for langchain-agentkit.

Each extension is a self-contained package under extensions/:

    from langchain_agentkit.extensions.skills import SkillsExtension
    from langchain_agentkit.extensions.agents import AgentsExtension
    from langchain_agentkit.extensions.tasks import TasksExtension
    from langchain_agentkit.extensions.teams import TeamExtension
    from langchain_agentkit.extensions.filesystem import FilesystemExtension
    from langchain_agentkit.extensions.hitl import HITLExtension
    from langchain_agentkit.extensions.web_search import WebSearchExtension
"""

from langchain_agentkit.extensions.agents import AgentsExtension
from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.extensions.history import HistoryExtension
from langchain_agentkit.extensions.hitl import HITLExtension
from langchain_agentkit.extensions.persistence import MessagePersistenceExtension
from langchain_agentkit.extensions.resilience import ResilienceExtension
from langchain_agentkit.extensions.skills import SkillsExtension
from langchain_agentkit.extensions.tasks import TasksExtension
from langchain_agentkit.extensions.teams import TeamExtension
from langchain_agentkit.extensions.web_search import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebSearchExtension,
)

__all__ = [
    "AgentsExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HITLExtension",
    "HistoryExtension",
    "MessagePersistenceExtension",
    "QwantSearchProvider",
    "ResilienceExtension",
    "SkillsExtension",
    "TasksExtension",
    "TeamExtension",
    "WebSearchExtension",
]
