"""Manual wiring — use AgentKit for full control over graph topology.

Use this approach when you need custom routing, multi-node graphs,
or a shared ToolNode. AgentKit composes tools, prompts, and state
schema from extensions.

For most cases, ``kit.compile(handler)`` builds the graph for you.
This example shows the manual-wiring path where you access kit
components (tools, prompt, state_schema) directly.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


# Pass user tools to AgentKit — kit.tools merges them with extension tools.
kit = AgentKit(
    extensions=[
        SkillsExtension(skills="skills/"),
        TasksExtension(),
    ],
    tools=[web_search],
)

llm = ChatOpenAI(model="gpt-4o")
all_tools = kit.tools  # includes web_search + extension tools
bound_llm = llm.bind_tools(all_tools)


async def researcher(state: dict) -> dict:
    """Research node that uses skills for methodology."""
    prompt = kit.prompt(state)
    messages = [SystemMessage(content=prompt)] + state["messages"]
    response = await bound_llm.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: dict) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# State schema composed from extensions (includes messages + tasks)
workflow = StateGraph(kit.state_schema)
workflow.add_node("researcher", researcher)
workflow.add_node("tools", ToolNode(all_tools))

workflow.add_edge(START, "researcher")
workflow.add_conditional_edges("researcher", should_continue, ["tools", END])
workflow.add_edge("tools", "researcher")

graph = workflow.compile()

if __name__ == "__main__":
    import asyncio

    result = asyncio.run(graph.ainvoke({"messages": [HumanMessage("Size the B2B SaaS market")]}))
    print(result["messages"][-1].content)
