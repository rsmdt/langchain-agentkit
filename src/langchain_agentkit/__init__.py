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
            result = await self.backend.read("AGENTS.md")
            return result.content or ""
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
from langchain_agentkit.agent_kit import AgentKit, run_extension_setup

# Backends — concrete backends are NOT re-exported here. Import each
# explicitly from its own submodule (langchain_agentkit.backends.os,
# langchain_agentkit.backends.daytona, …) so optional-dependency gates
# surface at the import line.
from langchain_agentkit.backends import BackendProtocol
from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent, wrap_if_needed
from langchain_agentkit.extension import Extension

# Extensions
from langchain_agentkit.extensions import (
    AgentsExtension,
    DuckDuckGoSearchProvider,
    FilesystemExtension,
    HistoryExtension,
    HITLExtension,
    MessagePersistenceExtension,
    QwantSearchProvider,
    ResilienceExtension,
    SkillsExtension,
    TasksExtension,
    TeamExtension,
    WebSearchExtension,
)

# Types
from langchain_agentkit.extensions.agents import AgentConfig
from langchain_agentkit.extensions.agents.filter import (
    DEFAULT_METADATA_PREFIX,
    strip_hidden_from_llm,
)
from langchain_agentkit.extensions.agents.output import (
    StrategyContext,
    SubagentOutput,
    SubagentOutputStrategy,
    full_history_strategy,
    last_message_strategy,
    resolve_output_strategy,
    trace_hidden_strategy,
)
from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.context_compaction import (
    CompactionSettings,
    ContextCompactionExtension,
)
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools
from langchain_agentkit.extensions.hitl import Option, Question
from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool
from langchain_agentkit.extensions.prompt_templates import (
    PromptTemplate,
    PromptTemplateExtension,
)
from langchain_agentkit.extensions.prompt_templates.tools import build_run_command_tool
from langchain_agentkit.extensions.skills import SkillConfig, build_skill_tool
from langchain_agentkit.extensions.tasks import Task, TasksState, TaskStatus, create_task_tools
from langchain_agentkit.extensions.teams import TeamState
from langchain_agentkit.extensions.teams.tools import create_team_tools
from langchain_agentkit.hook_runner import HookRunner
from langchain_agentkit.hooks import after, before, wrap

# Permissions
from langchain_agentkit.permissions import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
    PermissionRuleset,
)
from langchain_agentkit.prompt_composition import PromptComposition
from langchain_agentkit.state import AgentKitState

# Streaming
from langchain_agentkit.streaming import FilteredGraph, StreamingFilter

__all__ = [
    # Core
    "Agent",
    "AgentKit",
    "AgentKitState",
    "Extension",
    "HookRunner",
    "PromptComposition",
    "TasksState",
    "TeamState",
    "agent",
    "run_extension_setup",
    # Hook decorators
    "after",
    "before",
    "wrap",
    # Backends (capability protocol only; concrete backends live in submodules)
    "BackendProtocol",
    # Permissions
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    "PermissionRuleset",
    # Extensions
    "AgentsExtension",
    "CompactionSettings",
    "ContextCompactionExtension",
    "DuckDuckGoSearchProvider",
    "FilesystemExtension",
    "HITLExtension",
    "HistoryExtension",
    "MessagePersistenceExtension",
    "PromptTemplate",
    "PromptTemplateExtension",
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
    # Streaming
    "FilteredGraph",
    "StreamingFilter",
    # Types
    "AgentConfig",
    "SkillConfig",
    # Agents strategy API
    "DEFAULT_METADATA_PREFIX",
    "StrategyContext",
    "SubagentOutput",
    "SubagentOutputStrategy",
    "full_history_strategy",
    "last_message_strategy",
    "resolve_output_strategy",
    "strip_hidden_from_llm",
    "trace_hidden_strategy",
    # HITL types
    "Option",
    "Question",
    # Tools
    "Task",
    "TaskStatus",
    "build_run_command_tool",
    "build_skill_tool",
    "create_agent_tools",
    "create_ask_user_tool",
    "create_filesystem_tools",
    "create_task_tools",
    "create_team_tools",
]
