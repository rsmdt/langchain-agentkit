"""Root graph with checkpointer — enables interrupt() for multi-turn conversations.

When a node metaclass is the root graph (no parent), pass a checkpointer
to .compile() so interrupt() can persist state between turns.
"""

# ruff: noqa: N801, N805
import asyncio

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_agentkit import node


class advisor(node):
    llm = ChatOpenAI(model="gpt-4o")

    async def handler(state, *, llm):
        response = await llm.ainvoke(state["messages"])

        # No tool calls — ask the user for input instead of ending
        if not response.tool_calls:
            user_reply = interrupt(response.content)
            return {
                "messages": [response, HumanMessage(content=user_reply)],
            }

        return {"messages": [response]}


async def main():
    # Compile with a checkpointer for interrupt() support
    graph = advisor.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "session-1"}}

    # Turn 1: agent responds, hits interrupt() — graph pauses
    result = await graph.ainvoke(
        {"messages": [HumanMessage("Help me frame this problem")]},
        config,
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Check that the graph is paused, not ended
    state = await graph.aget_state(config)
    assert bool(state.tasks), "Expected pending interrupt"

    # Turn 2: resume with user input via Command(resume=...)
    result = await graph.ainvoke(
        Command(resume="B2B SaaS market declined 12% in Q3"),
        config,
    )
    print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
