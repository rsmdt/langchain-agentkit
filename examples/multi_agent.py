# ruff: noqa: N801, N805
"""Multi-agent graph — compose multiple node subclasses.

Each node metaclass produces a self-contained ReAct subgraph with its own
tools and skill access. Compose them in a parent graph for multi-agent workflows.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from langchain_agentkit import AgentState, node


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def sql_query(query: str) -> str:
    """Run a SQL query against the database."""
    return f"SQL results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))


class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "researcher"}


class analyst(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [sql_query, calculate]
    skills = "skills/"

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "sender": "analyst"}


# Compose in a parent graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher.compile())
workflow.add_node("analyst", analyst.compile())

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

graph = workflow.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage("Analyze the European SaaS market")]}
    )
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content[:100]}")
