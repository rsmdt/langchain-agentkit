# ruff: noqa: N801, N805
"""Subgraph checkpointer inheritance — interrupt() works without configuration.

When a node metaclass is used as a subgraph inside a parent graph,
it inherits the parent's checkpointer automatically. The subgraph
does not need its own checkpointer — just compile() with no arguments.
"""

import asyncio

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_agentkit import AgentState, node


class researcher(node):
    llm = ChatOpenAI(model="gpt-4o")

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])

        if not response.tool_calls:
            user_reply = interrupt(response.content)
            return {
                "messages": [response, HumanMessage(content=user_reply)],
            }

        return {"messages": [response]}


# Build parent graph — the parent owns the checkpointer
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher.compile())  # no checkpointer needed
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)

# Parent compiles with checkpointer — subgraph inherits it
parent = workflow.compile(checkpointer=InMemorySaver())


async def main():
    config = {"configurable": {"thread_id": "session-1"}}

    # Turn 1: subgraph's interrupt() works via inherited checkpointer
    result = await parent.ainvoke(
        {"messages": [HumanMessage("Research the European SaaS market")]},
        config,
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Turn 2: resume the subgraph
    result = await parent.ainvoke(
        Command(resume="Focus on B2B vertical SaaS"),
        config,
    )
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
