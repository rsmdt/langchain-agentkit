# ruff: noqa: N805
"""Web search — multi-provider search with zero config or custom providers.

WebSearchExtension fans out queries to all providers in parallel.
Works out of the box with built-in Qwant search (no API key needed).
Add your own providers for broader coverage.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import Agent, WebSearchExtension

# --- Example 1: Zero config (built-in Qwant search) ---


class QuickResearcher(Agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [WebSearchExtension()]
    prompt = "You are a research assistant. Use web search to find current information."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


# --- Example 2: Custom providers ---
# Uncomment to use Tavily (requires TAVILY_API_KEY):
#
# from langchain_tavily import TavilySearch
#
# class DeepResearcher(Agent):
#     model = ChatOpenAI(model="gpt-4o")
#     extensions = [WebSearchExtension(providers=[TavilySearch(max_results=5)])]
#     prompt = "You are a research assistant."
#
#     async def handler(state, *, llm, tools, prompt):
#         messages = [SystemMessage(content=prompt)] + state["messages"]
#         response = await llm.bind_tools(tools).ainvoke(messages)
#         return {"messages": [response]}


async def main():
    graph = QuickResearcher().compile()
    result = await graph.ainvoke(
        {"messages": [HumanMessage("What are the latest developments in AI agents?")]}
    )
    print(result["messages"][-1].content[:500])


if __name__ == "__main__":
    asyncio.run(main())
