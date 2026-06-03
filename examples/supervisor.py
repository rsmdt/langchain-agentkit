# ruff: noqa: N805
"""Supervisor pattern — embed an AgentKit agent as a *node* in a router graph.

An AgentKit ``Agent`` compiles to a normal LangGraph subgraph, so you can drop
it straight into a supervisor/router graph as a node. The agent shares the
parent's ``messages`` channel, and a tool you pass to the agent can return
``Command(goto=..., graph=Command.PARENT)`` to navigate the *parent* supervisor
directly. The hand-off signal is a real edge (the ``goto`` target) — not a tag
the supervisor has to sniff out of a message.

This needs **no** framework change: a hand-off tool is an ordinary tool.

    START -> router -> setup (embedded AgentKit agent)
      setup --(just answers)------------------> END             # path (a)
      setup --complete_phase: Command(advance)-> advance        # path (b)
      setup --return_to_supervisor: Command(...)-> classify_route  # path (c)

Note: a Command(graph=PARENT) escape propagates only its own ``update`` — the
agent's internal trace does not sync upward on its own. So the hand-off tool
forwards the agent's full turn trace (via InjectedState) to keep the
supervisor's message history complete and well-formed (no orphaned tool
messages). The wiring and this preservation behaviour are pinned by
tests/integration/test_supervisor_embedding.py (deterministic) and
tests/evals/test_supervisor_embedding_eval.py (LLM).

Run:  python examples/supervisor.py   (requires OPENAI_API_KEY)
"""

import asyncio
import operator
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

from langchain_agentkit import Agent


# --- 1. Hand-off-back tools: ordinary tools that return Command(graph=PARENT) ---
def make_handoff_tool(name: str, goto: str, description: str):
    """A tool that hands control back to the embedded agent's PARENT graph.

    Routes the parent to ``goto`` and forwards the agent's full turn trace (via
    InjectedState) into the shared ``messages`` channel, so the supervisor's
    history stays complete and well-formed for its next LLM call.
    """

    @tool(name, description=description)
    def _handoff(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        closing = ToolMessage(content=f"{name} -> {goto}", tool_call_id=tool_call_id)
        return Command(
            goto=goto,
            graph=Command.PARENT,  # escape the agent subgraph, route the parent
            # Forward the agent's paired AIMessage/ToolMessage trace; the original
            # HumanMessage is de-duplicated by add_messages (same id).
            update={"messages": [*state["messages"], closing]},
        )

    return _handoff


complete_phase = make_handoff_tool(
    "complete_phase", "advance", "Mark the setup phase complete and advance."
)
return_to_supervisor = make_handoff_tool(
    "return_to_supervisor",
    "classify_route",
    "Hand control back to the supervisor to route a non-setup request.",
)


@tool
def do_setup_work(detail: str) -> str:
    """Create part of the initial project scaffolding."""
    return f"created: {detail}"


# --- 2. The worker: a normal AgentKit Agent, embedded as a node below ---
class SetupAgent(Agent):
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = (
        "You handle the SETUP phase: creating initial project scaffolding.\n"
        "- If the request is setup work, do it with do_setup_work, then call "
        "complete_phase.\n"
        "- If the request is NOT about setup, call return_to_supervisor. Do not "
        "attempt work outside the setup phase."
    )
    tools = [do_setup_work, complete_phase, return_to_supervisor]
    max_turns = 6

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt), *state["messages"]]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


# --- 3. The supervisor graph that embeds the agent ---
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    visited: Annotated[list[str], operator.add]  # node trail, for illustration


def router(state: SupervisorState) -> dict:
    # Deterministic dispatch — real supervisors might read a DB / task queue.
    return {"visited": ["router"]}


def advance(state: SupervisorState) -> dict:
    return {"visited": ["advance"], "messages": [AIMessage(content="phase complete; advancing")]}


def classify_route(state: SupervisorState) -> dict:
    # In a real supervisor this is an LLM routing call that picks the next
    # node from the worker's hand-back message. Kept trivial here.
    return {"visited": ["classify_route"], "messages": [AIMessage(content="supervisor re-routing")]}


async def build_supervisor():
    setup = await SetupAgent().compile()  # AgentKit agent -> compiled subgraph

    sup = StateGraph(SupervisorState)
    sup.add_node("router", router)
    sup.add_node("setup", setup)  # ← embed the AgentKit agent as a node
    sup.add_node("advance", advance)
    sup.add_node("classify_route", classify_route)

    sup.add_edge(START, "router")
    sup.add_edge("router", "setup")
    sup.add_edge("setup", END)  # path (a): agent answered, no hand-off fired
    sup.add_edge("advance", END)
    sup.add_edge("classify_route", END)

    return sup.compile()


async def main() -> None:
    supervisor = await build_supervisor()

    print("== on-phase request (expect hand-off -> advance) ==")
    result = await supervisor.ainvoke(
        {"messages": [HumanMessage("Scaffold the project: create a src/ dir and a pyproject.toml")]}
    )
    print("route:", result["visited"])
    print("final:", result["messages"][-1].content)

    print("\n== off-phase request (expect hand-back -> classify_route) ==")
    result = await supervisor.ainvoke(
        {"messages": [HumanMessage("What were last quarter's sales figures?")]}
    )
    print("route:", result["visited"])
    print("final:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
