# ruff: noqa: N805
"""Standalone agent — the simplest way to use langchain-agentkit.

Declare a class with the Agent base class and get a complete ReAct agent
with extension support. Agent.graph() returns an uncompiled StateGraph
(for composition), Agent.compile() returns a compiled runnable.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import Agent, SkillsExtension


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


class Researcher(Agent):
    model = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    extensions = [SkillsExtension(skills="skills/")]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


# Researcher().compile() returns a compiled runnable directly
if __name__ == "__main__":
    graph = Researcher().compile()
    result = graph.invoke({"messages": [HumanMessage("Size the B2B SaaS market in Europe")]})
    print(result["messages"][-1].content)
