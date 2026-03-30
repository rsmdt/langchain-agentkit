"""Composable extension framework for LangGraph agents.

Two layers:

**Primitive** — use ``AgentKit`` for full graph control::

    from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension

    kit = AgentKit(extensions=[SkillsExtension("skills/"), TasksExtension()])
    all_tools = my_tools + kit.tools
    prompt = kit.prompt(state, runtime)

**Convenience** — use ``agent`` metaclass for standalone ReAct agents::

    from langchain_agentkit import agent, SkillsExtension

    class researcher(agent):
        llm = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension("skills/")]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, prompt):
            messages = [SystemMessage(content=prompt)] + state["messages"]
            return {"messages": [await llm.ainvoke(messages)]}

    graph = researcher.compile()

**Standalone** — use ``SkillRegistry`` directly::

    from langchain_agentkit import SkillRegistry

    registry = SkillRegistry("skills/")
    tools = registry.tools  # [Skill]
"""

from langgraph.prebuilt import ToolRuntime

# Core
from langchain_agentkit.agent import agent
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.hooks import after, before, wrap

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
from langchain_agentkit.state import (
    AgentKitState,
    TasksState,
    TeamState,
)

# Tools
from langchain_agentkit.tools import (
    SkillRegistry,
    Task,
    TaskStatus,
    create_filesystem_tools,
    create_task_tools,
)

# Composability
from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent, wrap_if_needed

# Backend
from langchain_agentkit.backend import (
    BackendProtocol,
    CompositeBackend,
    LocalBackend,
    MemoryBackend,
    SandboxProtocol,
)

# VFS
from langchain_agentkit.vfs import VirtualFilesystem

__all__ = [
    # Core
    "AgentKit",
    "AgentKitState",
    "Extension",
    "TasksState",
    "TeamState",
    "ToolRuntime",
    "VirtualFilesystem",
    "agent",
    # Hook decorators
    "after",
    "before",
    "wrap",
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
    # Backend
    "BackendProtocol",
    "CompositeBackend",
    "LocalBackend",
    "MemoryBackend",
    "SandboxProtocol",
    # Tools
    "SkillRegistry",
    "Task",
    "TaskStatus",
    "create_filesystem_tools",
    "create_task_tools",
]
