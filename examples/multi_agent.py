# ruff: noqa: N805
"""Multi-agent graph — compose multiple Agent subclasses.

Each Agent subclass produces a self-contained ReAct subgraph with its own
tools and extensions. Compose them in a parent graph for multi-agent workflows.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from langchain_agentkit import Agent, AgentKit, SkillsExtension


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
    return str(eval(expression))  # noqa: S307


class Researcher(Agent):
    model = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    extensions = [SkillsExtension(skills="skills/")]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


class Analyst(Agent):
    model = ChatOpenAI(model="gpt-4o")
    tools = [sql_query, calculate]
    prompt = "You are a data analyst."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


# Compose in a parent graph — use AgentKit to get the combined state schema
kit = AgentKit(extensions=[SkillsExtension(skills="skills/")])

workflow = StateGraph(kit.state_schema)
workflow.add_node("researcher", Researcher().compile())
workflow.add_node("analyst", Analyst().compile())

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

graph = workflow.compile()

if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage("Analyze the European SaaS market")]})
    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content[:100]}")
