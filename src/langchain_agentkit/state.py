"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Minimal state for multi-agent graphs.

    Extend with your own fields::

        class MyState(AgentState):
            my_custom_field: str

    Fields:
        messages: Conversation history with LangGraph message aggregation.
        sender: Name of the last node that called tools (for routing back).
    """

    messages: Annotated[list[Any], add_messages]
    sender: str
