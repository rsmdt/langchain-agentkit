# ruff: noqa: N801, N805
"""Standalone agent — the simplest way to use langchain-agentkit.

Declare a class with the agent metaclass and get a complete ReAct agent
with extension support. The result is a StateGraph — call .compile() to run it.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import SkillsExtension, agent


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


class researcher(agent):
    model = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    extensions = [SkillsExtension(skills="skills/")]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


# researcher is a StateGraph — compile and invoke
if __name__ == "__main__":
    graph = researcher.compile()
    result = graph.invoke({"messages": [HumanMessage("Size the B2B SaaS market in Europe")]})
    print(result["messages"][-1].content)
