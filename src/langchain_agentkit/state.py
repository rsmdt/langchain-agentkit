"""Composable state schemas for LangGraph agents.

``AgentKitState`` is the minimal base — just messages, sender, and jump_to.
Extensions add state keys via mixins (e.g., ``TasksState``, ``TeamState``).

Usage::

    from langchain_agentkit.state import AgentKitState, TasksState

    # Compose manually
    class MyState(AgentKitState, TasksState):
        my_field: str

    # Or let AgentKit compose from extensions automatically
    kit = AgentKit([TasksExtension()])
    kit.state_schema  # → composed TypedDict with messages + tasks
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _last_writer_wins(left: str | None, right: str | None) -> str | None:
    """Reducer that takes the most recent value (right wins)."""
    return right


class AgentKitState(TypedDict, total=False):
    """Minimal base state — always present in any agentkit graph.

    Contains only the fields required for the ReAct loop to function.
    Extension adds additional keys via mixin TypedDicts.
    """

    messages: Annotated[list[Any], add_messages]
    sender: str
    _agentkit_jump_to: Annotated[str | None, _last_writer_wins]


# Re-export extension state schemas from their canonical locations.
# This preserves backward compat for `from langchain_agentkit.state import TasksState`.
from langchain_agentkit.extensions.tasks.state import TasksState as TasksState  # noqa: E402
from langchain_agentkit.extensions.teams.state import TeamState as TeamState  # noqa: E402
