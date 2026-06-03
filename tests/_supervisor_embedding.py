"""Shared helpers for the supervisor-embedding scenario.

Used by both the deterministic wiring test
(``tests/integration/test_supervisor_embedding.py``) and the LLM-behaviour eval
(``tests/evals/test_supervisor_embedding_eval.py``) so the topology and the
hand-off tools stay identical across both.

The scenario: an AgentKit agent is embedded as a *node* inside a supervisor
graph, sharing the parent's ``messages`` channel. Tools passed to the agent
return ``Command(goto=..., graph=Command.PARENT)`` to navigate the parent
supervisor directly — the hand-off signal is a real edge, not a sniffed message.
No AgentKit change is required: a hand-off tool is an ordinary tool.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

# Prompt shared by the deterministic and LLM builds so behaviour expectations
# match the wiring the deterministic test pins down.
SETUP_PROMPT = (
    "You handle the SETUP phase: creating initial project scaffolding "
    "(directories, config files, boilerplate).\n"
    "- If the user's request is about setup work, do it with do_setup_work, "
    "then call complete_phase to advance.\n"
    "- If the request is NOT about setup (a different phase or off-topic), call "
    "return_to_supervisor so the supervisor can route it elsewhere. Do not "
    "attempt work outside the setup phase."
)


def make_handoff_tool(name: str, goto: str, description: str) -> Any:
    """Build a tool that hands control back to the embedded agent's PARENT graph.

    Returns a Command that (a) routes the parent to *goto* and (b) forwards the
    agent's full turn trace into the parent's shared ``messages`` channel, closed
    by this tool's own ToolMessage.

    Forwarding the trace via ``InjectedState`` is what keeps the parent history
    complete AND well-formed. A ``Command(graph=PARENT)`` escape otherwise
    propagates ONLY its own ``update`` — the agent's internal (AIMessage,
    ToolMessage) pairs never sync upward, which both loses the trace and orphans
    this closing ToolMessage (its calling AIMessage stays in the subgraph),
    breaking any downstream supervisor LLM call. It is still a plain tool — the
    framework needs no special support for it.
    """

    @tool(name, description=description)
    def _handoff(
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[dict, InjectedState],
    ) -> Command:
        closing = ToolMessage(content=f"{name}->{goto}", tool_call_id=tool_call_id)
        return Command(
            goto=goto,
            graph=Command.PARENT,
            # Forward the agent's full turn trace (paired AIMessage/ToolMessage
            # sequence) so the parent history stays complete and valid. The
            # original HumanMessage is de-duplicated by add_messages (same id).
            update={"messages": [*state["messages"], closing]},
        )

    return _handoff


def orphaned_tool_messages(messages: list[BaseMessage]) -> list[str]:
    """Return tool_call_ids of ToolMessages with no preceding AIMessage tool call.

    A non-empty result means the message history is malformed — a downstream LLM
    call over these messages would be rejected (every tool message must follow an
    assistant message bearing the matching tool_call_id).
    """
    called_ids = {tc["id"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    return [
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and m.tool_call_id not in called_ids
    ]


@tool
def do_setup_work(detail: str) -> str:
    """Perform a unit of setup work."""
    return f"did:{detail}"


def handoff_tools() -> list[Any]:
    """The two hand-off-back tools the embedded agent carries."""
    return [
        make_handoff_tool("complete_phase", "advance", "Mark this phase complete and advance."),
        make_handoff_tool(
            "return_to_supervisor",
            "classify_route",
            "Hand control back to the supervisor to pick a different phase.",
        ),
    ]


class SupervisorState(TypedDict):
    """Parent graph state: shared messages + a visited-node trail for assertions."""

    messages: Annotated[list, add_messages]
    visited: Annotated[list[str], operator.add]


def build_supervisor(embedded_agent: Any, checkpointer: Any = None) -> Any:
    """Compile a supervisor graph that embeds *embedded_agent* as the 'setup' node.

    START -> router -> setup (embedded AgentKit agent)
      setup --static-------------------------> END             (path a: answered)
      setup --Command(goto='advance', PARENT)--> advance -> END (path b: complete)
      setup --Command(goto='classify_route')---> classify_route -> END (path c: hand back)

    Pass *checkpointer* to persist the thread on the parent only; the embedded
    subgraph inherits it (do not compile the subgraph with its own checkpointer).
    """

    def router(state: SupervisorState) -> dict:
        return {"visited": ["router"]}

    def advance(state: SupervisorState) -> dict:
        return {"visited": ["advance"], "messages": [AIMessage(content="advanced")]}

    def classify_route(state: SupervisorState) -> dict:
        return {
            "visited": ["classify_route"],
            "messages": [AIMessage(content="re-dispatched")],
        }

    sup = StateGraph(SupervisorState)
    sup.add_node("router", router)
    sup.add_node("setup", embedded_agent)  # ← AgentKit agent embedded as a node
    sup.add_node("advance", advance)
    sup.add_node("classify_route", classify_route)

    sup.add_edge(START, "router")
    sup.add_edge("router", "setup")
    # Default exit when the agent just answers (no hand-off tool fired) — the
    # diagram's path (a). Hand-off tools override this via Command(goto=...).
    sup.add_edge("setup", END)
    sup.add_edge("advance", END)
    sup.add_edge("classify_route", END)

    return sup.compile(checkpointer=checkpointer)
