"""Custom state type — use a different state shape via handler annotation.

Annotate the handler's first parameter to use a custom TypedDict as the
graph's state type. Custom fields survive graph execution. Without an
annotation, the state schema is composed from middleware automatically.
"""

# ruff: noqa: N801, N805
import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from langchain_agentkit import agent

# --- Define a custom state with domain-specific fields ---


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    draft: dict | None
    components_saved: list[str]


# --- Define tools ---


@tool
def save_component(name: str, content: str) -> str:
    """Save a component to the draft."""
    return f'{{"status": "saved", "component": "{name}"}}'


# --- Declare an agent with custom state ---


class drafter(agent):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [save_component]

    async def handler(state: WorkflowState, *, llm):
        messages = [SystemMessage(content="You are a document drafter.")] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}


async def main():
    graph = drafter.compile()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage("Draft the introduction")],
            "draft": {"intro": "initial content"},
            "components_saved": ["intro"],
        }
    )

    # Custom fields are preserved
    print(f"Draft: {result.get('draft')}")
    print(f"Components: {result.get('components_saved')}")
    print(f"Last message: {result['messages'][-1].content[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
