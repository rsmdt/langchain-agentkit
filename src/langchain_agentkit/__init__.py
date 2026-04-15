"""Composable extension framework for LangGraph agents.

**Declarative** — the ``Agent`` class builds a ReAct graph from class attributes::

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension(skills="skills/")]
        prompt = "You are a research assistant."
        async def handler(state, *, llm, tools, prompt): ...

    app = Researcher().compile()           # compiled runnable
    graph = Researcher().graph()           # uncompiled StateGraph (for composition)

**Dynamic** — properties can be sync/async methods for per-request resolution::

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        async def prompt(self):
            return await self.backend.read("AGENTS.md")
        async def handler(state, *, llm, tools, prompt): ...

    app = Researcher(backend=my_backend).compile()

**Primitive** — ``AgentKit`` for managed or manual graph wiring::

    kit = AgentKit(
        extensions=[SkillsExtension(skills="skills/"), TasksExtension()],
        model=ChatOpenAI(model="gpt-4o"),
    )
    graph = kit.compile(handler)      # managed ReAct loop
    # or access kit.tools, kit.compose(), kit.model directly
"""

# Core
from langchain_agentkit.agent import Agent, agent
from langchain_agentkit.agent_kit import AgentKit

# Backends
from langchain_agentkit.backends import (
    BackendProtocol,
    DaytonaBackend,
    OSBackend,
)
from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent, wrap_if_needed
from langchain_agentkit.extension import Extension

# Extensions
from langchain_agentkit.extensions import (
    AgentsExtension,
    DuckDuckGoSearchProvider,
    FilesystemExtension,
    HistoryExtension,
    HITLExtension,
    QwantSearchProvider,
    ResilienceExtension,
    SkillsExtension,
    TasksExtension,
    TeamExtension,
    WebSearchExtension,
)

# Types
from langchain_agentkit.extensions.agents import AgentConfig
from langchain_agentkit.extensions.hitl import Option, Question
from langchain_agentkit.extensions.skills import SkillConfig, build_skill_tool
from langchain_agentkit.extensions.tasks import Task, TasksState, TaskStatus, create_task_tools
from langchain_agentkit.extensions.teams import TeamState
from langchain_agentkit.hooks import after, before, wrap

# Permissions
from langchain_agentkit.permissions import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
    PermissionRuleset,
)
from langchain_agentkit.state import AgentKitState

__all__ = [
    # Core
    "Agent",
    "AgentKit",
    "AgentKitState",
    "Extension",
    "TasksState",
    "TeamState",
    "agent",
    # Hook decorators
    "after",
    "before",
    "wrap",
    # Backends
    "BackendProtocol",
    "DaytonaBackend",
    "OSBackend",
    # Permissions
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    "PermissionRuleset",
    # Extensions
    "AgentsExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HistoryExtension",
    "HITLExtension",
    "QwantSearchProvider",
    "ResilienceExtension",
    "SkillsExtension",
    "TasksExtension",
    "TeamExtension",
    "WebSearchExtension",
    # Composability
    "AgentLike",
    "CompiledAgent",
    "TeamAgent",
    "wrap_if_needed",
    # Types
    "AgentConfig",
    "SkillConfig",
    # HITL types
    "Option",
    "Question",
    # Tools
    "Task",
    "TaskStatus",
    "build_skill_tool",
    "create_task_tools",
]
