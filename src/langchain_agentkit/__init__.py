"""Composable extension framework for LangGraph agents.

**Primitive** — use ``AgentKit`` for full graph control::

    from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension

    kit = AgentKit(extensions=[SkillsExtension("skills/"), TasksExtension()])
    all_tools = my_tools + kit.tools

**Convenience** — use ``agent`` metaclass for standalone ReAct agents::

    from langchain_agentkit import agent, SkillsExtension

    class researcher(agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension("skills/")]
        prompt = "You are a research assistant."
        async def handler(state, *, llm, prompt):
            ...

    graph = researcher.compile()
"""

# Core
from langchain_agentkit.agent import agent
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent, wrap_if_needed
from langchain_agentkit.extension import Extension
from langchain_agentkit.hooks import after, before, wrap

# Backends
from langchain_agentkit.backends import (
    BackendProtocol,
    BaseSandbox,
    DaytonaSandbox,
    OSBackend,
)

# Permissions
from langchain_agentkit.permissions import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
    PermissionRuleset,
)

# Extensions
from langchain_agentkit.extensions import (
    AgentExtension,
    DuckDuckGoSearchProvider,
    FilesystemExtension,
    HITLExtension,
    QwantSearchProvider,
    SkillsExtension,
    TasksExtension,
    TeamExtension,
    WebSearchExtension,
)

# Types
from langchain_agentkit.extensions.agents import AgentConfig
from langchain_agentkit.extensions.hitl import Option, Question
from langchain_agentkit.extensions.skills import SkillConfig, build_skill_tool
from langchain_agentkit.extensions.tasks import Task, TaskStatus, TasksState, create_task_tools
from langchain_agentkit.extensions.teams import TeamState
from langchain_agentkit.state import AgentKitState

__all__ = [
    # Core
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
    "BaseSandbox",
    "DaytonaSandbox",
    "OSBackend",
    # Permissions
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    "PermissionRuleset",
    # Extensions
    "AgentExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HITLExtension",
    "QwantSearchProvider",
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
