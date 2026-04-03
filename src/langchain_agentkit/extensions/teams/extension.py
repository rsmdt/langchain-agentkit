"""TeamExtension — message-driven team coordination."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.extensions.teams.bus import (
        ActiveTeam,
        TeamMessage,
    )

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_team_coordination_template = PromptTemplate.from_file(_PROMPT_FILE)


class TeamExtension(Extension):
    """Extension providing message-driven team coordination.

    Args:
        agents: List of StateGraph objects with ``.name`` and ``.description``.
        ephemeral: Enable dynamic (on-the-fly) team agents.
        max_team_size: Maximum number of team members allowed.
        router_timeout: Seconds to wait for messages in the Router Node.
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        ephemeral: bool = False,
        max_team_size: int = 5,
        router_timeout: float = 30.0,
    ) -> None:
        from langchain_agentkit.extensions.agents.extension import _validate_agent_list

        if max_team_size < 1:
            raise ValueError("max_team_size must be >= 1")

        self._agents_by_name: dict[str, Any] = _validate_agent_list(agents)
        self._ephemeral = ephemeral
        self._max_team_size = max_team_size
        self._router_timeout = router_timeout
        self._active_team: ActiveTeam | None = None
        self._team_lock: asyncio.Lock = asyncio.Lock()
        self._parent_llm_getter: Any = None

        from langchain_agentkit.extensions.teams.tools import create_team_tools

        self._tools = tuple(create_team_tools(self))

    def set_parent_llm_getter(self, getter: Any) -> None:
        self._parent_llm_getter = getter

    # --- Public accessors for tool implementations ---

    @property
    def active_team(self) -> ActiveTeam | None:
        return self._active_team

    @active_team.setter
    def active_team(self, value: ActiveTeam | None) -> None:
        self._active_team = value

    @property
    def max_team_size(self) -> int:
        return self._max_team_size

    @property
    def ephemeral(self) -> bool:
        return self._ephemeral

    @property
    def agents_by_name(self) -> dict[str, Any]:
        return self._agents_by_name

    @property
    def parent_llm_getter(self) -> Any:
        return self._parent_llm_getter

    @property
    def team_lock(self) -> asyncio.Lock:
        return self._team_lock

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools  # type: ignore[return-value]

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> str:
        roster_lines = []
        for name, graph in self._agents_by_name.items():
            desc = getattr(graph, "description", "")
            roster_lines.append(f"- **{name}**: {desc}" if desc else f"- **{name}**")
        agent_roster = "\n".join(roster_lines)
        base_prompt = _team_coordination_template.format(agent_roster=agent_roster)

        if self._active_team is not None:
            team = self._active_team
            status_lines = [f"\n### Active Team: {team.name}\n"]
            for name, task in team.members.items():
                agent_type = team.member_types.get(name, "unknown")
                if task.done():
                    try:
                        task.result()
                        icon = "✅"
                    except asyncio.CancelledError:
                        icon = "🚫"
                    except Exception:
                        icon = "❌"
                else:
                    icon = "🔄"
                pending = team.bus.pending_count(name)
                pending_str = f" ({pending} pending)" if pending > 0 else ""
                status_lines.append(f"- {icon} **{name}** ({agent_type}){pending_str}")
            lead_pending = team.bus.pending_count("lead")
            if lead_pending > 0:
                status_lines.append(
                    f"\n⚠️ You have **{lead_pending} unread message(s)**. "
                    "Use TeamStatus to collect them."
                )
            base_prompt += "\n".join(status_lines)

        return base_prompt

    def dependencies(self) -> list[Any]:
        from langchain_agentkit.extensions.tasks.extension import TasksExtension

        return [TasksExtension()]

    @property
    def state_schema(self) -> type:
        from langchain_agentkit.extensions.teams.state import TeamState

        return TeamState

    def graph_modifier(self, workflow: Any, node_name: str) -> Any:  # noqa: C901
        """Inject the Router Node into the graph topology."""
        from langgraph.graph import END

        mw = self

        async def _drain_messages(team: Any) -> list[TeamMessage]:
            """Drain pending messages; fall back to blocking receive."""
            msgs: list[TeamMessage] = []
            lead_queue = team.bus._queues.get("lead")
            if lead_queue is not None:
                while not lead_queue.empty():
                    try:
                        msgs.append(lead_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
            if not msgs:
                active = sum(1 for t in team.members.values() if not t.done())
                if active == 0:
                    return msgs
                msg = await team.bus.receive("lead", timeout=mw._router_timeout)
                if msg is not None:
                    msgs.append(msg)
            return msgs

        async def _router_node(state: dict[str, Any]) -> dict[str, Any]:
            from langchain_agentkit.extensions.teams.task_router import (
                classify_and_process,
            )

            team = mw._active_team
            if team is None:
                return {}

            raw_messages = await _drain_messages(team)
            if not raw_messages:
                return {}

            tasks = list(state.get("tasks") or [])
            return await classify_and_process(raw_messages, tasks, team.bus)

        def _router_should_continue(state: dict[str, Any]) -> str:
            team = mw._active_team
            if team is None:
                msgs = state.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if hasattr(last, "type") and last.type == "tool":
                        return node_name
                return END
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                kwargs = getattr(last, "additional_kwargs", {})
                if kwargs.get("type") == "teammate_message":
                    return node_name
            active_count = sum(1 for t in team.members.values() if not t.done())
            lead_pending = team.bus.pending_count("lead")
            if active_count == 0 and lead_pending == 0:
                return END
            return node_name

        workflow.add_node("router", _router_node)
        workflow.add_conditional_edges(
            "router",
            _router_should_continue,
            {node_name: node_name, END: END},
        )

        return workflow
