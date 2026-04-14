"""Shared schemas, guards, and helpers for team tools."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Annotated, Any

from langchain_core.tools import InjectedToolCallId, ToolException
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

# Pydantic needs these at runtime for Field annotations on _AgentSpec.
from langchain_agentkit.extensions.agents.refs import Dynamic, Predefined  # noqa: TC001

if TYPE_CHECKING:
    from langchain_agentkit.extensions.teams.extension import TeamExtension


# ---------------------------------------------------------------------------
# Teammate system prompt addendum
# ---------------------------------------------------------------------------

_TEAMMATE_ADDENDUM = """\

# Agent Teammate Communication

IMPORTANT: You are running as an agent in a team. To communicate with anyone \
on your team:
- Use the TeamMessage tool with `to: "<name>"` to send messages to specific teammates
- Use the TeamMessage tool with `to: "*"` sparingly for team-wide broadcasts

Just writing a response in text is not visible to others on your team — \
you MUST use the TeamMessage tool.

# Task Management

You share a task list with your team. Use the standard task tools to coordinate:
- **TaskList** to see available work
- **TaskCreate** to add new tasks
- **TaskUpdate** to claim tasks (set owner), update status, or mark completed
- **TaskGet** to read full task details
"""


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _AgentSpec(BaseModel):
    """Specification for a single team agent."""

    name: str = Field(description="Unique name for this agent in the team.")
    agent: Predefined | Dynamic = Field(
        description=(
            "Agent to use. Use {id: name} to select a pre-defined agent "
            "from the roster, or {prompt: text} to create an ephemeral "
            "reasoning agent for this team role."
        ),
    )


class _TeamCreateInput(BaseModel):
    name: str = Field(description="Name for the team.")
    agents: list[_AgentSpec] = Field(
        description=(
            "List of agents to include. Each needs a unique name and "
            "an agent reference ({id} or {prompt})."
        ),
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamMessageInput(BaseModel):
    to: str = Field(
        description='Recipient: teammate name, or "*" for broadcast to all teammates.',
    )
    message: str = Field(description="Message content.")
    summary: str = Field(
        default="",
        description="5-10 word summary shown as preview (optional).",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamStatusInput(BaseModel):
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


class _TeamDissolveInput(BaseModel):
    timeout: float = Field(
        default=30.0,
        description="Max seconds to wait for graceful shutdown.",
    )
    state: Annotated[dict[str, Any], InjectedState]
    tool_call_id: Annotated[str, InjectedToolCallId]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_active_team(ext: TeamExtension) -> Any:
    """Return the active team or raise ToolException."""
    team = ext.active_team
    if team is None:
        raise ToolException("No active team. Call TeamCreate first.")
    return team


def _require_member(ext: TeamExtension, member_name: str) -> None:
    """Verify member exists in the active team."""
    team = _require_active_team(ext)
    if member_name not in team.members:
        raise ToolException(f"Member '{member_name}' not in active team.")


# ---------------------------------------------------------------------------
# Predefined agent proxy compilation
# ---------------------------------------------------------------------------

_TASK_TOOL_NAMES = frozenset({"TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskStop"})


def _compile_with_proxy_tasks(
    agent_graph: Any,
    bus: Any,
    member_name: str,
) -> Any:
    """Compile a predefined agent graph, replacing task tools with proxies."""
    handler = getattr(agent_graph, "_agentkit_handler", None)
    llm = getattr(agent_graph, "_agentkit_llm", None)
    kit = getattr(agent_graph, "_agentkit_kit", None)
    original_tools = getattr(agent_graph, "_agentkit_user_tools", None)

    if handler is None or llm is None or kit is None:
        return agent_graph.compile()

    from langchain_agentkit.extensions.teams.task_proxy import create_task_proxy_tools

    proxy_tools = create_task_proxy_tools(bus, member_name)
    filtered_user_tools = [t for t in (original_tools or []) if t.name not in _TASK_TOOL_NAMES]
    new_user_tools = filtered_user_tools + proxy_tools

    from langchain_agentkit.agent_kit import AgentKit
    from langchain_agentkit.extensions.tasks import TasksExtension

    filtered_extensions = [ext for ext in kit.extensions if not isinstance(ext, TasksExtension)]
    graph_name = getattr(agent_graph, "name", member_name)
    new_kit = AgentKit(
        extensions=filtered_extensions,
        prompt=kit.base_prompt,
        tools=new_user_tools,
        model=llm,
        name=graph_name,
    )

    rebuilt = new_kit.compile(handler)
    return rebuilt.compile()


# ---------------------------------------------------------------------------
# Bus / task cleanup helpers (also used by extension.py)
# ---------------------------------------------------------------------------


async def _shutdown_team_tasks(
    team: Any,
    timeout: float,
) -> None:
    """Send structured shutdown requests and wait for tasks to finish."""
    shutdown_msg = json.dumps({"type": "shutdown_request"})
    for member_name in team.members:
        with contextlib.suppress(Exception):
            await team.bus.send("lead", member_name, shutdown_msg)

    pending_tasks = [t for t in team.members.values() if not t.done()]
    if pending_tasks:
        _done, still_pending = await asyncio.wait(pending_tasks, timeout=timeout)
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.wait(still_pending, timeout=5.0)


def _cleanup_bus(team: Any) -> None:
    """Unregister all agents from the message bus."""
    for name in list(team.members.keys()):
        with contextlib.suppress(Exception):
            team.bus.unregister(name)
    with contextlib.suppress(Exception):
        team.bus.unregister("lead")


def _unassign_teammate_tasks(
    tasks: list[dict[str, Any]],
    member_names: list[str],
) -> list[dict[str, Any]]:
    """Reset unresolved tasks owned by dissolved teammates to pending."""
    name_set = set(member_names)
    result = []
    for t in tasks:
        t = dict(t)
        owner = t.get("owner")
        if owner and owner in name_set and t.get("status") not in ("completed", "deleted"):
            t["status"] = "pending"
            t["owner"] = None
        result.append(t)
    return result
