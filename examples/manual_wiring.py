"""Manual wiring — use SkillRegistry as a standalone toolkit.

Use this approach when you want full control over your LangGraph graph
and just need the Skill + SkillRead tools added to your tool list.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit import AgentState, SkillRegistry


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


llm = ChatOpenAI(model="gpt-4o")
kit = SkillRegistry("skills/")

# Combine your tools with skill tools
all_tools = [web_search] + kit.tools
bound_llm = llm.bind_tools(all_tools)


async def researcher(state: AgentState) -> dict:
    """Research node that uses skills for methodology."""
    response = await bound_llm.ainvoke(state["messages"])
    return {"messages": [response], "sender": "researcher"}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# Build the graph manually
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher)
workflow.add_node("tools", ToolNode(all_tools))

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", should_continue, ["tools", END])
workflow.add_edge("tools", "researcher")

graph = workflow.compile()

if __name__ == "__main__":
    import asyncio

    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage("Size the B2B SaaS market")]})
    )
    print(result["messages"][-1].content)
