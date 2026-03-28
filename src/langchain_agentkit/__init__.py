"""Composable middleware framework for LangGraph agents.

Two layers:

**Primitive** — use ``AgentKit`` for full graph control::

    from langchain_agentkit import AgentKit, SkillsMiddleware, TasksMiddleware

    kit = AgentKit([SkillsMiddleware("skills/"), TasksMiddleware()])
    all_tools = my_tools + kit.tools
    prompt = kit.prompt(state, runtime)

**Convenience** — use ``agent`` metaclass for standalone ReAct agents::

    from langchain_agentkit import agent, SkillsMiddleware

    class researcher(agent):
        llm = ChatOpenAI(model="gpt-4o")
        middleware = [SkillsMiddleware("skills/")]
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

# Middleware
from langchain_agentkit.middleware import (
    AgentMiddleware,
    AgentTeamMiddleware,
    DuckDuckGoSearchProvider,
    FilesystemMiddleware,
    HITLMiddleware,
    Middleware,
    QwantSearchProvider,
    SkillsMiddleware,
    TasksMiddleware,
    WebSearchMiddleware,
)
from langchain_agentkit.state import (
    AgentKitState,
    AgentState,
    SubAgentState,
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

# VFS
from langchain_agentkit.vfs import VirtualFilesystem

__all__ = [
    # Core
    "AgentKit",
    "AgentKitState",
    "AgentState",
    "Middleware",
    "SubAgentState",
    "TasksState",
    "TeamState",
    "ToolRuntime",
    "VirtualFilesystem",
    "agent",
    # Middleware
    "AgentMiddleware",
    "AgentTeamMiddleware",
    "FilesystemMiddleware",
    "HITLMiddleware",
    "SkillsMiddleware",
    "TasksMiddleware",
    "WebSearchMiddleware",
    # Tools
    "DuckDuckGoSearchProvider",
    "QwantSearchProvider",
    "SkillRegistry",
    "Task",
    "TaskStatus",
    "create_filesystem_tools",
    "create_task_tools",
]
