# ruff: noqa: N801, N805
"""Standalone node — the simplest way to use langchain-agentkit.

Declare a class with the node metaclass and get a complete ReAct agent
with skill support. The result is a StateGraph — call .compile() to run it.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import node


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "researcher"}


# researcher is a StateGraph — compile and invoke
if __name__ == "__main__":
    graph = researcher.compile()
    result = graph.invoke(
        {"messages": [HumanMessage("Size the B2B SaaS market in Europe")]}
    )
    print(result["messages"][-1].content)
