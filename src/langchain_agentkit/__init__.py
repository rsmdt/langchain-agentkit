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
        model = ChatOpenAI(model="gpt-4o")
        extensions = [SkillsExtension("skills/")]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, prompt):
            messages = [SystemMessage(content=prompt)] + state["messages"]
            return {"messages": [await llm.ainvoke(messages)]}

    graph = researcher.compile()

**Standalone** — use ``build_skill_tool`` directly::

    from langchain_agentkit import SkillConfig, build_skill_tool

    configs = [SkillConfig(name="research", description="...", prompt="...")]
    tool = build_skill_tool(configs)  # Skill tool
"""

from langgraph.prebuilt import ToolRuntime

# Core
from langchain_agentkit.agent import agent
from langchain_agentkit.agent_kit import AgentKit

# Backend
from langchain_agentkit.backend import (
    BackendProtocol,
    OSBackend,
    SandboxProtocol,
)

# Composability
from langchain_agentkit.composability import AgentLike, CompiledAgent, TeamAgent, wrap_if_needed
from langchain_agentkit.extension import Extension

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
from langchain_agentkit.hooks import after, before, wrap
from langchain_agentkit.state import (
    AgentKitState,
    TasksState,
    TeamState,
)

# Tools
from langchain_agentkit.tools import (
    Task,
    TaskStatus,
    build_skill_tool,
    create_filesystem_tools,
    create_task_tools,
)
from langchain_agentkit.types import SkillConfig

__all__ = [
    # Core
    "AgentKit",
    "AgentKitState",
    "Extension",
    "TasksState",
    "TeamState",
    "ToolRuntime",
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
    "OSBackend",
    "SandboxProtocol",
    # Types
    "AgentConfig",
    "SkillConfig",
    # Tools
    "Task",
    "TaskStatus",
    "build_skill_tool",
    "create_filesystem_tools",
    "create_task_tools",
]
