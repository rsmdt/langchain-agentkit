"""Composable extension framework for LangGraph agents.

``Agent.graph()`` and ``Agent.compile()`` are async — extension setup
and dynamic property resolution may issue backend I/O. Callers must
``await`` them.

**Declarative** — the ``Agent`` class builds a ReAct graph from class attributes::

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension(skills="skills/")]
        prompt = "You are a research assistant."
        async def handler(state, *, llm, tools, prompt): ...

    app = await Researcher().compile()           # compiled runnable
    graph = await Researcher().graph()           # uncompiled StateGraph (for composition)

**Dynamic** — properties can be sync/async methods for per-request resolution::

    class Researcher(Agent):
        model = ChatOpenAI(model="gpt-4o")
        async def prompt(self):
            result = await self.backend.read("AGENTS.md")
            return result.content or ""
        async def handler(state, *, llm, tools, prompt): ...

    app = await Researcher(backend=my_backend).compile()

**Primitive** — ``AgentKit`` for managed or manual graph wiring. Composition
is sync; use ``run_extension_setup`` to await async extension setup::

    kit = AgentKit(
        extensions=[SkillsExtension(skills="skills/"), TasksExtension()],
        model=ChatOpenAI(model="gpt-4o"),
    )
    await run_extension_setup(kit)    # async setup (backend discovery, etc.)
    graph = kit.compile(handler)      # managed ReAct loop (sync)
    # or access kit.tools, kit.compose(), kit.model directly
"""

# Core
# Backends — concrete backends are NOT re-exported here. Import each
# explicitly from its own submodule (langchain_agentkit.backends.os,
# langchain_agentkit.backends.daytona, langchain_agentkit.backends.agentfs,
# …) so optional-dependency gates surface at the import line.
from langchain_agentkit.backends import (
    EditError,
    EditResult,
    ExecuteResponse,
    FileDownloadResult,
    FileError,
    FilesystemProtocol,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    SandboxProtocol,
    WriteResult,
)
from langchain_agentkit.composition.agent import Agent
from langchain_agentkit.composition.agent_kit import AgentKit, run_extension_setup
from langchain_agentkit.composition.composability import (
    AgentLike,
    CompiledAgent,
    TeamAgent,
    wrap_if_needed,
)
from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.composition.hooks import after, before, wrap
from langchain_agentkit.composition.prompts import PromptComposition
from langchain_agentkit.composition.state import AgentKitState

# Extensions
from langchain_agentkit.extensions import (
    AgentsExtension,
    CoreBehaviorExtension,
    DuckDuckGoSearchProvider,
    EnvExtension,
    FilesystemExtension,
    HistoryExtension,
    HITLExtension,
    MemoryExtension,
    MessagePersistenceExtension,
    QwantSearchProvider,
    ResilienceExtension,
    SkillsExtension,
    TasksExtension,
    TeamExtension,
    TurnBudgetExtension,
    WebSearchExtension,
)

# Types
from langchain_agentkit.extensions.agents import AgentConfig
from langchain_agentkit.extensions.agents.output_strategies import (
    StrategyContext,
    SubagentOutput,
    SubagentOutputStrategy,
    full_history_strategy,
    last_message_strategy,
    resolve_output_strategy,
    trace_hidden_strategy,
)
from langchain_agentkit.extensions.agents.tools import create_agent_tools
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools
from langchain_agentkit.extensions.history import (
    CompactionStrategy,
    CountStrategy,
    HistoryStrategy,
    TokenStrategy,
)
from langchain_agentkit.extensions.hitl import Option, Question
from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool
from langchain_agentkit.extensions.skills import SkillConfig, build_skill_tool
from langchain_agentkit.extensions.tasks import Task, TasksState, TaskStatus, create_task_tools
from langchain_agentkit.extensions.teams import TeamState
from langchain_agentkit.extensions.teams.tools import create_team_tools
from langchain_agentkit.extensions.turn_budget import TurnBudgetState

# Permissions
from langchain_agentkit.permissions import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
    PermissionRuleset,
)

__all__ = [
    # Core
    "Agent",
    "AgentKit",
    "AgentKitState",
    "Extension",
    "PromptComposition",
    "TasksState",
    "TeamState",
    "TurnBudgetState",
    "run_extension_setup",
    # Hook decorators
    "after",
    "before",
    "wrap",
    # Backends (capability protocols and result types; concrete backends live in submodules)
    "EditError",
    "EditResult",
    "ExecuteResponse",
    "FileDownloadResult",
    "FileError",
    "FileUploadResult",
    "FilesystemProtocol",
    "ReadBytesResult",
    "ReadResult",
    "SandboxEnvironment",
    "SandboxProtocol",
    "WriteResult",
    # Permissions
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    "PermissionRuleset",
    # Extensions
    "AgentsExtension",
    "CompactionStrategy",
    "CoreBehaviorExtension",
    "CountStrategy",
    "DuckDuckGoSearchProvider",
    "EnvExtension",
    "FilesystemExtension",
    "HITLExtension",
    "HistoryExtension",
    "HistoryStrategy",
    "MemoryExtension",
    "MessagePersistenceExtension",
    "QwantSearchProvider",
    "ResilienceExtension",
    "SkillsExtension",
    "TasksExtension",
    "TeamExtension",
    "TokenStrategy",
    "TurnBudgetExtension",
    "WebSearchExtension",
    # Composability
    "AgentLike",
    "CompiledAgent",
    "TeamAgent",
    "wrap_if_needed",
    # Types
    "AgentConfig",
    "SkillConfig",
    # Agents strategy API
    "StrategyContext",
    "SubagentOutput",
    "SubagentOutputStrategy",
    "full_history_strategy",
    "last_message_strategy",
    "resolve_output_strategy",
    "trace_hidden_strategy",
    # HITL types
    "Option",
    "Question",
    # Tools
    "Task",
    "TaskStatus",
    "build_skill_tool",
    "create_agent_tools",
    "create_ask_user_tool",
    "create_filesystem_tools",
    "create_task_tools",
    "create_team_tools",
]
