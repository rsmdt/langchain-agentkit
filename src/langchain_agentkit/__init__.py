"""Composable middleware framework for LangGraph agents.

Two layers:

**Primitive** — use ``AgentKit`` for full graph control::

    from langchain_agentkit import AgentKit, SkillsMiddleware, TasksMiddleware

    kit = AgentKit([SkillsMiddleware("skills/"), TasksMiddleware()])
    all_tools = my_tools + kit.tools
    prompt = kit.prompt(state, config)

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

**Standalone toolkit** — use ``SkillKit`` directly::

    from langchain_agentkit import SkillKit

    kit = SkillKit("skills/")
    tools = kit.tools  # [Skill, SkillRead]
"""

from langchain_agentkit.agent import agent
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.middleware import Middleware
from langchain_agentkit.node import node
from langchain_agentkit.skill_kit import SkillKit
from langchain_agentkit.skills_middleware import SkillsMiddleware
from langchain_agentkit.state import AgentState
from langchain_agentkit.tasks_middleware import TasksMiddleware

__all__ = [
    # Primitive
    "AgentKit",
    "Middleware",
    # Convenience
    "agent",
    "node",
    # Standalone
    "SkillKit",
    # Middleware implementations
    "SkillsMiddleware",
    "TasksMiddleware",
    # State
    "AgentState",
]
