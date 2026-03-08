"""Custom state type — use a different state shape via handler annotation.

Annotate the handler's first parameter to use a custom TypedDict as the
graph's state type. Custom fields survive graph execution. Without an
annotation, AgentState is used by default.
"""

# ruff: noqa: N801, N805
import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from langchain_agentkit import node

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


class drafter(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [save_component]

    async def handler(state: WorkflowState, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}


# --- Different state shape: subgraph with schema translation ---


class ParentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    project_name: str
    final_draft: dict | None


class child_agent(node):
    llm = ChatOpenAI(model="gpt-4o")

    async def handler(state: WorkflowState, *, llm):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}


def call_child(state: ParentState) -> ParentState:
    """Wrapper that translates between parent and child state shapes."""
    # Map parent state → child state
    child_input: WorkflowState = {
        "messages": state["messages"],
        "draft": state.get("final_draft"),
        "components_saved": [],
    }

    graph = child_agent.compile()
    result = graph.invoke(child_input)

    # Map child state → parent state
    return {
        "messages": result["messages"],
        "final_draft": result.get("draft"),
    }


async def main():
    # Example 1: Custom state fields survive execution
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
