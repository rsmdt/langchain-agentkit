"""TeamExtension — message-driven team coordination with cross-turn rehydration."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import FilesystemProtocol
    from langchain_agentkit.extensions.teams.bus import (
        ActiveTeam,
        TeamMessage,
        TeamMessageBus,
    )
    from langchain_agentkit.extensions.teams.state import TeammateSpec

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_team_coordination_template = PromptTemplate.from_file(_PROMPT_FILE)
_logger = logging.getLogger("langchain_agentkit.extensions.teams")


# Sentinel default for ``_parent_tools_getter``: identity-checked in
# ``setup()`` so kit-level wiring only overrides the default, not a
# user-supplied getter. Returns ``[]`` so unwired call sites stay safe.
def _default_tools_getter() -> list[Any]:
    return []


class TeamExtension(Extension):
    """Extension providing message-driven team coordination.

    Team state survives across HTTP request/response turns via two
    persisted channels:

    * ``state["team"]`` — ``TeamMetadata`` (name, member specs).  Enough
      to rebuild the runtime on any pod.
    * ``state["messages"]`` — shared conversation with teammate-internal
      messages tagged ``additional_kwargs["team"]["member"]``.  The lead's
      ``wrap_model`` filter hides them from the lead model call; each
      teammate's rehydration rebuilds its history from the filtered
      slice.

    Runtime (bus, ``asyncio.Task``s, compiled teammate graphs, capture
    buffer) is per-turn only — created at turn start via
    ``_rehydrate_if_needed`` and torn down at turn end via ``after_run``.

    Three input modes for ``agents``:

    * **List** — programmatic objects (StateGraph / AgentLike / AgentConfig).
    * **Path** — a string or Path to a directory scanned for agent
      ``.md`` files via :func:`discover_agents_from_directory`.
    * **Path + backend** — discovery deferred to async ``setup()`` and
      performed via :func:`discover_agents_from_backend`.

    Config-based teammates are compiled on-demand at team-spawn time
    using the kit's ``model_resolver``, the lead's tool roster (filtered
    by ``AgentConfig.tools``) and a sibling ``SkillsExtension`` (if
    present, for ``AgentConfig.skills`` resolution).

    Args:
        agents: Member roster — see modes above.
        backend: Optional :class:`FilesystemProtocol` for async directory
            discovery.
        ephemeral: Enable dynamic (on-the-fly) team agents.
        max_team_size: Maximum number of team members allowed.
        router_timeout: Seconds to wait for messages in the Router Node.
        max_history_tokens: Per-teammate history budget applied at rebuild
            time via ``trim_messages``.  Does not modify state; only caps
            the history passed into each teammate's compiled graph.
        token_counter: Passed through to ``trim_messages``.  Use
            ``"approximate"`` for the dependency-free heuristic, or pass
            an LLM instance for model-accurate counting.
        tools: Optional explicit tool list. When ``None`` (default), the
            extension builds its standard set (TeamCreate, TeamMessage,
            TeamStatus, TeamDissolve) — these close over the extension
            instance to access bus state and the active team. When
            provided, these tools replace the defaults entirely; the user
            is responsible for any closure wiring their replacements need.
    """

    def __init__(
        self,
        *,
        agents: list[Any] | str | Path,
        backend: FilesystemProtocol | None = None,
        ephemeral: bool = False,
        max_team_size: int = 5,
        router_timeout: float = 30.0,
        max_history_tokens: int = 20_000,
        token_counter: Any = "approximate",
        tools: Sequence[BaseTool] | None = None,
    ) -> None:
        from langchain_agentkit.extensions.agents.refs import validate_agent_list
        from langchain_agentkit.extensions.agents.types import (
            _AgentConfigProxy,
            _wrap_agents,
        )

        if max_team_size < 1:
            raise ValueError("max_team_size must be >= 1")

        self._backend = backend
        self._deferred_path: str | None = None

        if isinstance(agents, list):
            wrapped = _wrap_agents(agents)
            self._agents_by_name: dict[str, Any] = validate_agent_list(wrapped)
        elif isinstance(agents, (str, Path)):
            if backend is not None:
                self._deferred_path = str(agents)
                self._agents_by_name = {}
            else:
                from langchain_agentkit.extensions.agents.discovery import (
                    discover_agents_from_directory,
                )

                defs = discover_agents_from_directory(Path(agents))
                proxies = [_AgentConfigProxy(d) for d in defs]
                self._agents_by_name = validate_agent_list(proxies) if proxies else {}
        else:
            raise TypeError(f"agents must be list, str, or Path, got {type(agents).__name__}")

        self._ephemeral = ephemeral
        self._max_team_size = max_team_size
        self._router_timeout = router_timeout
        self._max_history_tokens = max_history_tokens
        self._token_counter: Any = token_counter
        self._active_team: ActiveTeam | None = None
        self._team_lock: asyncio.Lock = asyncio.Lock()
        self._parent_llm_getter: Any = None
        self._parent_tools_getter: Any = _default_tools_getter
        self._model_resolver: Any = None
        self._skills_resolver: Any = None
        # Per-turn mutable list populated by teammate loops, drained by
        # the before_model hook.  A new list is allocated on each turn
        # during rehydration so the tasks spawned on that turn bind to
        # the correct object.
        self._capture_buffer: list[BaseMessage] = []

        if tools is not None:
            self._tools: tuple[BaseTool, ...] = tuple(tools)
        else:
            from langchain_agentkit.extensions.teams.tools import create_team_tools

            self._tools = tuple(create_team_tools(self))

    def set_parent_llm_getter(self, getter: Any) -> None:
        self._parent_llm_getter = getter

    def set_parent_tools_getter(self, getter: Any) -> None:
        self._parent_tools_getter = getter

    @property
    def model_resolver(self) -> Callable[[str], BaseChatModel] | None:
        return self._model_resolver  # type: ignore[no-any-return]

    @property
    def parent_tools_getter(self) -> Any:
        return self._parent_tools_getter

    @property
    def skills_resolver(self) -> Any:
        return self._skills_resolver

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
    def capture_buffer(self) -> list[BaseMessage]:
        return self._capture_buffer

    @property
    def max_history_tokens(self) -> int:
        return self._max_history_tokens

    @property
    def token_counter(self) -> Any:
        return self._token_counter

    @property
    @override
    def tools(self) -> list[BaseTool]:
        return self._tools  # type: ignore[return-value]

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str | dict[str, str]:
        roster_lines = []
        for name, graph in self._agents_by_name.items():
            desc = getattr(graph, "description", "")
            roster_lines.append(f"- **{name}**: {desc}" if desc else f"- **{name}**")
        agent_roster = "\n".join(roster_lines)
        # Static: coordination guidance + the fixed agent roster -> cacheable
        # system prompt. Live team status is dynamic per turn -> reminder.
        base_prompt = _team_coordination_template.format(agent_roster=agent_roster)

        # The only per-turn signal the lead must act on is unread messages from
        # teammates; live member status is one TeamStatus call away. Mirror the
        # system prompt's term ("Agent Team") and phrase it neutrally.
        if self._active_team is not None:
            pending = self._active_team.bus.pending_count("lead")
            if pending:
                verb, noun, pron = (
                    ("is", "message", "it") if pending == 1 else ("are", "messages", "them")
                )
                return {
                    "prompt": base_prompt,
                    "reminder": (
                        f"There {verb} {pending} unread {noun} from the Agent Team. "
                        f"Use TeamStatus to collect {pron}."
                    ),
                }

        return base_prompt

    @override
    def dependencies(self) -> list[Any]:
        from langchain_agentkit.extensions.tasks.extension import TasksExtension

        return [TasksExtension()]

    @override
    async def setup(  # type: ignore[override]
        self,
        *,
        extensions: list[Extension],
        model_resolver: Any = None,
        llm_getter: Any = None,
        tools_getter: Any = None,
        **_: Any,
    ) -> None:
        from langchain_agentkit.extensions.history.extension import HistoryExtension

        # --- Sibling ordering check ---
        # TeamExtension must precede any HistoryExtension so its wrap_model
        # filter runs outermost. If History ran first it would truncate a view
        # still containing team-tagged messages and orphan the lead's tool-call
        # pairs.
        my_index: int | None = None
        history_index: int | None = None
        for i, ext in enumerate(extensions):
            if ext is self:
                my_index = i
            elif isinstance(ext, HistoryExtension) and history_index is None:
                history_index = i
        if my_index is not None and history_index is not None and history_index < my_index:
            raise ValueError(
                "TeamExtension must be listed before HistoryExtension in the "
                "AgentKit extensions list so its wrap_model filter runs outermost. "
                f"TeamExtension is at index {my_index}, HistoryExtension at {history_index}."
            )

        # --- Pick up kit-level model_resolver ---
        if model_resolver is not None and self._model_resolver is None:
            self._model_resolver = model_resolver

        # --- Wire kit-level parent LLM / tools getters ---
        # Ephemeral teammates and config-based teammates without an
        # explicit ``model`` inherit the parent LLM via
        # ``parent_llm_getter``. Tools-inheriting teammates resolve their
        # parent tool set via ``parent_tools_getter``.
        if llm_getter is not None and self._parent_llm_getter is None:
            self._parent_llm_getter = llm_getter
        if tools_getter is not None and self._parent_tools_getter is _default_tools_getter:
            self._parent_tools_getter = tools_getter

        # --- Deferred backend discovery ---
        await self._run_deferred_discovery()

        # --- Skills resolver (for AgentConfig.skills on config-based teammates) ---
        self._wire_skills_resolver(extensions)

    async def _run_deferred_discovery(self) -> None:
        if self._deferred_path is None or self._backend is None:
            return
        from langchain_agentkit.extensions.agents.discovery import (
            discover_agents_from_backend,
        )
        from langchain_agentkit.extensions.agents.refs import validate_agent_list
        from langchain_agentkit.extensions.agents.types import _AgentConfigProxy

        defs = await discover_agents_from_backend(self._backend, self._deferred_path)
        proxies = [_AgentConfigProxy(d) for d in defs]
        self._agents_by_name = validate_agent_list(proxies) if proxies else {}
        self._deferred_path = None

    def _wire_skills_resolver(self, extensions: list[Extension]) -> None:
        from langchain_agentkit.extensions.skills import SkillsExtension

        skills_ext = next(
            (e for e in extensions if isinstance(e, SkillsExtension)),
            None,
        )
        if skills_ext is None:
            return
        configs = skills_ext.configs

        def _resolve_skills(
            names: list[str],
            _configs: list[Any] = configs,
        ) -> str:
            index = {c.name: c for c in _configs}
            parts = []
            for name in names:
                config = index.get(name)
                if config:
                    parts.append(config.prompt)
            return "\n\n".join(parts)

        self._skills_resolver = _resolve_skills

    @property
    @override
    def state_schema(self) -> type:
        from langchain_agentkit.extensions.teams.state import TeamState

        return TeamState

    def graph_modifier(self, workflow: Any, node_name: str) -> Any:  # noqa: C901
        from langgraph.graph import END

        from langchain_agentkit.extensions.teams.filter import is_team_tagged

        mw = self

        # Route terminating edges to ``_run_exit`` when run-lifecycle hooks
        # are wired into the graph; otherwise jump straight to END.  This
        # relies on ``graph_builder`` adding ``_run_exit`` before invoking
        # ``graph_modifier``.  Without this, a router→END transition would
        # skip ``after_run`` cleanup (teammate tasks, bus, capture buffer).
        # ``workflow.nodes`` is a dict on real ``StateGraph``; test mocks
        # may omit it, in which case we fall back to END.
        workflow_nodes = getattr(workflow, "nodes", None) or {}
        terminal: str = "_run_exit" if "_run_exit" in workflow_nodes else END

        async def _drain_messages(team: Any) -> list[TeamMessage]:
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
                return terminal
            msgs = state.get("messages", [])
            if msgs:
                last = msgs[-1]
                if hasattr(last, "type") and last.type == "human" and not is_team_tagged(last):
                    return node_name
            active_count = sum(1 for t in team.members.values() if not t.done())
            lead_pending = team.bus.pending_count("lead")
            if active_count == 0 and lead_pending == 0:
                return terminal
            return node_name

        workflow.add_node("router", _router_node)
        workflow.add_conditional_edges(
            "router",
            _router_should_continue,
            {node_name: node_name, terminal: terminal},
        )

        return workflow

    # ------------------------------------------------------------------
    # Teammate graph construction
    # ------------------------------------------------------------------

    def build_teammate_graph(self, spec: TeammateSpec, bus: TeamMessageBus) -> Any:
        from langchain_core.tools import ToolException

        from langchain_agentkit.composability import AgentLike
        from langchain_agentkit.extensions.agents.types import AgentConfig
        from langchain_agentkit.extensions.teams.task_proxy import create_task_proxy_tools
        from langchain_agentkit.extensions.teams.tools.shared import (
            _TEAMMATE_ADDENDUM,
            _compile_config_with_proxy_tasks,
            _compile_with_proxy_tasks,
        )
        from langchain_agentkit.graph_builder import build_ephemeral_graph

        member_name = spec["member_name"]
        kind = spec["kind"]

        if kind == "predefined":
            agent_id = spec["agent_id"]
            target = self._agents_by_name[agent_id]  # raises KeyError if missing
            config = getattr(target, "_agent_config", None)
            if isinstance(config, AgentConfig):
                return _compile_config_with_proxy_tasks(
                    config,
                    bus,
                    member_name,
                    parent_tools_getter=self._parent_tools_getter,
                    parent_llm_getter=self._parent_llm_getter,
                    model_resolver=self._model_resolver,
                    skills_resolver=self._skills_resolver,
                )
            if isinstance(target, AgentLike):
                return target
            return _compile_with_proxy_tasks(target, bus, member_name)

        if kind == "dynamic":
            system_prompt = spec["system_prompt"]
            if self._parent_llm_getter is None:
                raise ToolException(
                    "Cannot build dynamic teammate — no parent LLM getter wired. "
                    "AgentKit wires this at graph build time."
                )
            llm = self._parent_llm_getter()
            return build_ephemeral_graph(
                name=member_name,
                llm=llm,
                prompt=_TEAMMATE_ADDENDUM + system_prompt,
                user_tools=create_task_proxy_tools(bus, member_name),
                checkpointer=None,
            )

        raise ValueError(f"Unknown TeammateSpec kind: {kind!r}")

    # ------------------------------------------------------------------
    # Rehydration
    # ------------------------------------------------------------------

    async def rehydrate_if_needed(self, state: dict[str, Any]) -> None:
        # Idempotent within a turn: the first call wins; later calls (parallel
        # tool bodies or the before_model hook) see ``_active_team`` set and
        # return. No-op when state carries no team metadata.
        team_meta = state.get("team") if isinstance(state, dict) else None
        if team_meta is None:
            return
        if self._active_team is not None:
            return

        async with self._team_lock:
            # Re-check under the lock: another tool body may have
            # rehydrated between our first check and lock acquisition.
            if self._active_team is not None:
                return
            await self._do_rehydrate(team_meta, state)

    async def _do_rehydrate(  # noqa: C901
        self,
        team_meta: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        # Inner rehydration — must be called under ``_team_lock``.
        from typing import cast

        from langchain_core.messages import trim_messages

        from langchain_agentkit.extensions.teams.bus import (
            ActiveTeam,
            TeamMessageBus,
            _teammate_loop,
        )
        from langchain_agentkit.extensions.teams.filter import filter_team_messages

        team_name: str = team_meta.get("name", "")
        members: list[TeammateSpec] = [
            cast("TeammateSpec", m) for m in (team_meta.get("members") or [])
        ]

        _logger.info(
            "Rehydrating team %r with %d member(s)",
            team_name,
            len(members),
            extra={"team_name": team_name, "member_count": len(members)},
        )

        # Fresh bus and capture buffer — binding for newly-spawned tasks.
        bus = TeamMessageBus()
        bus.register("lead")
        self._capture_buffer = []

        all_messages = list(state.get("messages") or [])

        member_tasks: dict[str, asyncio.Task[str]] = {}
        member_types: dict[str, str] = {}

        for spec in members:
            member_name = spec["member_name"]
            bus.register(member_name)

            raw_history = filter_team_messages(all_messages, member_name)
            capped_history = trim_messages(
                raw_history,
                max_tokens=self._max_history_tokens,
                token_counter=self._token_counter,
                strategy="last",
                allow_partial=False,
                include_system=False,
            )

            try:
                compiled = self.build_teammate_graph(spec, bus)
            except KeyError as exc:
                _logger.warning(
                    "Missing roster agent during rehydration: %s",
                    exc,
                    extra={
                        "team_name": team_name,
                        "member_name": member_name,
                        "agent_id": spec.get("agent_id"),
                    },
                )
                member_types[member_name] = f"unavailable:{spec.get('agent_id', '?')}"
                member_tasks[member_name] = await _make_degraded_task(member_name)
                continue
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    "Failed to rebuild teammate %r: %s",
                    member_name,
                    exc,
                    extra={"team_name": team_name, "member_name": member_name},
                )
                member_types[member_name] = "unavailable:build_failed"
                member_tasks[member_name] = await _make_degraded_task(member_name)
                continue

            task = asyncio.create_task(
                _teammate_loop(
                    member_name,
                    compiled,
                    bus,
                    initial_history=list(capped_history),
                    capture_buffer=self._capture_buffer,
                ),
                name=f"team-{team_name}-{member_name}",
            )
            member_tasks[member_name] = task
            member_types[member_name] = _type_label_for_spec(spec)

        self._active_team = ActiveTeam(
            name=team_name,
            bus=bus,
            members=member_tasks,
            member_types=member_types,
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def before_model(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        await self.rehydrate_if_needed(state)
        if not self._capture_buffer:
            return None
        drained = list(self._capture_buffer)
        self._capture_buffer.clear()
        return {"messages": drained}

    async def wrap_model(
        self,
        *,
        state: dict[str, Any],
        handler: Any,
        runtime: Any,
    ) -> Any:
        # Non-destructive: checkpointed state keeps everything; only the
        # inner handler sees the team-tagged messages filtered out.
        from langchain_agentkit.extensions.teams.filter import filter_out_team_messages

        original = list(state.get("messages") or [])
        filtered = filter_out_team_messages(original)
        filtered_state = {**state, "messages": filtered}
        return await handler(filtered_state)

    async def after_run(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        # Returns only the message flush — does NOT set ``team: None``. Team
        # metadata persists across turns so rehydration works; only
        # ``TeamDissolve`` clears it.
        from langchain_agentkit.extensions.teams.tools.shared import (
            _cleanup_bus,
            _shutdown_team_tasks,
        )

        remaining = list(self._capture_buffer)
        self._capture_buffer.clear()

        if self._active_team is not None:
            _logger.info(
                "Turn-end cleanup for team %r with %d member(s)",
                self._active_team.name,
                len(self._active_team.members),
                extra={
                    "team_name": self._active_team.name,
                    "member_count": len(self._active_team.members),
                    "buffer_size": len(remaining),
                },
            )
            try:
                await _shutdown_team_tasks(self._active_team, timeout=30.0)
            except Exception:  # noqa: BLE001
                _logger.exception("Error during teammate shutdown")
            _cleanup_bus(self._active_team)
            self._active_team = None

        if remaining:
            return {"messages": remaining}
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _type_label_for_spec(spec: TeammateSpec) -> str:
    kind = spec.get("kind")
    if kind == "predefined":
        return spec.get("agent_id", "predefined")
    if kind == "dynamic":
        return f"ephemeral:{spec['member_name']}"
    return "unknown"


# An already-completed failed task representing a degraded slot (a
# rehydration target that couldn't be rebuilt). The slot stays visible in
# ``active_team.members`` so ``task_status`` reports "failed" and any
# ``TeamMessage(to=...)`` surfaces a clear ``ToolException``.
async def _make_degraded_task(member_name: str) -> asyncio.Task[str]:
    async def _degraded() -> str:
        raise RuntimeError(f"unavailable:{member_name}")

    import contextlib

    task: asyncio.Task[str] = asyncio.create_task(_degraded())
    # Drive the coroutine to completion so ``task.done()`` is True on
    # return.  We swallow the synthetic exception here; ``task.result()``
    # will still re-raise it on inspection, which is exactly how
    # ``task_status`` detects the "failed" state.
    with contextlib.suppress(RuntimeError):
        await task
    return task
