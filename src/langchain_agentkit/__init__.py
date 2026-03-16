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
    tools = registry.tools  # [Skill, SkillRead]
"""

from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.agent import agent
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.hitl_middleware import HITLMiddleware
from langchain_agentkit.middleware import Middleware
from langchain_agentkit.skill_registry import SkillRegistry
from langchain_agentkit.skills_middleware import SkillsMiddleware
from langchain_agentkit.state import AgentState
from langchain_agentkit.task_tools import Task, TaskStatus, create_task_tools
from langchain_agentkit.tasks_middleware import TasksMiddleware
from langchain_agentkit.web_search_middleware import QwantSearchTool, WebSearchMiddleware

__all__ = [
    # Primitive
    "AgentKit",
    "Middleware",
    "ToolRuntime",
    # Convenience
    "agent",
    # Standalone
    "SkillRegistry",
    # Middleware implementations
    "HITLMiddleware",
    "SkillsMiddleware",
    "TasksMiddleware",
    "WebSearchMiddleware",
    # Tools
    "QwantSearchTool",
    # Task tools
    "create_task_tools",
    # Types
    "Task",
    "TaskStatus",
    # State
    "AgentState",
]
