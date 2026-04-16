"""AgentsExtension — blocking subagent delegation with parallel support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.agents.discovery import (
    discover_agents_from_directory,
)
from langchain_agentkit.extensions.agents.refs import validate_agent_list
from langchain_agentkit.extensions.agents.types import (
    _AgentConfigProxy,
    _wrap_agents,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import BackendProtocol

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_agent_delegation_template = PromptTemplate.from_file(_PROMPT_FILE)


def _default_tools_getter() -> list[Any]:
    """Sentinel default for ``_parent_tools_getter``.

    Identity-checked in ``setup()`` so kit-level wiring only overrides
    the default — not a user-supplied getter that happens to equal
    ``list``. Returns ``[]`` so unwired call sites stay safe.
    """
    return []


_CONCISENESS_DIRECTIVE = (
    "\n\nWhen reporting delegation results, be concise. "
    "Synthesize the key findings — don't repeat the subagent's full response verbatim."
)

_DYNAMIC_SECTION = (
    "\nEphemeral custom agents are supported — provide a system prompt "
    "instead of a roster id. Ephemeral agents are reasoning-only and "
    "cannot use tools."
)

_PARALLEL_NOTE = "\nMultiple Agent calls issued in the same turn run concurrently."


def _get_tools_description(agent: Any) -> str:
    """Return a human-readable summary of an agent's tool restrictions.

    Reads ``_agent_config`` (if present) to determine allowed/disallowed
    tools and formats them for the roster prompt.

    Returns:
        ``"*"`` (all tools) when no restrictions are configured,
        ``"All tools except X, Y"`` for denylist-only,
        ``"X, Y, Z"`` for allowlist-only, or the effective filtered
        list when both are present.
    """
    from langchain_agentkit.extensions.agents.types import AgentConfig

    config: AgentConfig | None = getattr(agent, "_agent_config", None)
    if config is None:
        return "*"

    allowlist: list[str] | None = getattr(config, "tools", None)
    denylist: list[str] | None = getattr(config, "disallowed_tools", None)

    has_allow = allowlist is not None and len(allowlist) > 0
    has_deny = denylist is not None and len(denylist) > 0

    if not has_allow and not has_deny:
        return "All tools"

    if has_deny and not has_allow:
        return f"All tools except {', '.join(sorted(denylist))}"  # type: ignore[arg-type]

    if has_allow and not has_deny:
        return ", ".join(sorted(allowlist))  # type: ignore[arg-type]

    # Both present — effective set is allowlist minus denylist
    effective = sorted(set(allowlist) - set(denylist))  # type: ignore[arg-type]
    if not effective:
        return "No tools"
    return ", ".join(effective)


class AgentsExtension(Extension):
    """Extension providing blocking subagent delegation via the Agent tool.

    Two input modes:
    - **List**: Pass agent objects (StateGraph, AgentLike, or AgentConfig).
    - **Path**: Pass a string or Path to a directory to scan for .md files.

    Args:
        agents: List of agent objects, or a string/Path to a directory.
        backend: Optional BackendProtocol for remote filesystem discovery.
        ephemeral: Enable dynamic (on-the-fly) agents.
        default_conciseness: Append conciseness directive.
        delegation_timeout: Max seconds to wait for a subagent response.
        output_mode: Strategy that shapes the subagent's result when
            merged into the parent graph's messages list. Accepts a
            built-in name (``"last_message"`` / ``"full_history"`` /
            ``"trace_hidden"``) or a callable with the signature
            ``(SubagentOutput, StrategyContext) -> list[BaseMessage]``.
            Defaults to ``"trace_hidden"``, which persists every
            subagent AIMessage (tagged hidden-from-LLM) plus a plain
            final ``ToolMessage``. See
            :mod:`langchain_agentkit.extensions.agents.output` for the
            strategy API and built-in shapes, and
            :class:`HideSubagentTraceExtension` for the paired filter.
        metadata_prefix: Namespace for the ``response_metadata`` tag
            keys the strategy writes and the filter reads. Defaults to
            ``"agentkit"``.
    """

    def __init__(
        self,
        *,
        agents: list[Any] | str | Path,
        backend: BackendProtocol | None = None,
        ephemeral: bool = False,
        default_conciseness: bool = True,
        delegation_timeout: float = 300.0,
        output_mode: Any = "trace_hidden",
        metadata_prefix: str = "agentkit",
    ) -> None:
        from langchain_agentkit.extensions.agents.output import (
            StrategyContext,
            resolve_output_strategy,
            trace_hidden_strategy,
        )

        self._backend = backend
        self._deferred_path: str | None = None
        self._output_strategy = resolve_output_strategy(output_mode)
        self._metadata_prefix = metadata_prefix
        self._strategy_context = StrategyContext(metadata_prefix=metadata_prefix)
        # True when the resolved strategy tags subagent messages with
        # ``{prefix}_hidden_from_llm=True``. Drives both the wrap_model
        # filter and the setup() ordering check. Only the built-in
        # ``trace_hidden_strategy`` does this today; custom strategies
        # that tag messages should either reuse it or be documented to
        # set a ``_tags_hidden_from_llm = True`` attribute on themselves.
        self._strategy_tags_hidden = (
            self._output_strategy is trace_hidden_strategy
            or bool(getattr(self._output_strategy, "_tags_hidden_from_llm", False))
        )

        if isinstance(agents, list):
            wrapped = _wrap_agents(agents)
            self._agents_by_name: dict[str, Any] = validate_agent_list(wrapped)
            self._has_config_agents = any(isinstance(a, _AgentConfigProxy) for a in wrapped)
        elif isinstance(agents, (str, Path)):
            if backend is not None:
                # Defer async discovery to setup()
                self._deferred_path = str(agents)
                self._agents_by_name = {}
                self._has_config_agents = False
            else:
                defs = discover_agents_from_directory(Path(agents))
                proxies = [_AgentConfigProxy(d) for d in defs]
                if proxies:
                    self._agents_by_name = validate_agent_list(proxies)
                else:
                    self._agents_by_name = {}
                self._has_config_agents = bool(proxies)
        else:
            msg = f"agents must be list, str, or Path, got {type(agents).__name__}"
            raise TypeError(msg)

        self._ephemeral = ephemeral
        self._default_conciseness = default_conciseness
        self._delegation_timeout = delegation_timeout
        self._compiled_cache: dict[str, Any] = {}

        self._parent_tools_getter: Any = _default_tools_getter
        self._parent_llm_getter: Any = None
        self._model_resolver: Any = None
        self._skills_resolver: Any = None

        self._tools = tuple(self._create_tools())

    @property
    def model_resolver(self) -> Callable[[str], BaseChatModel] | None:
        """Return the configured model resolver, if any.

        Exposed as a public attribute so ``AgentKit.resolve_model()`` can
        discover it without any cross-extension wiring.
        """
        return self._model_resolver  # type: ignore[no-any-return]

    async def setup(  # type: ignore[override]
        self,
        *,
        extensions: list[Extension],
        model_resolver: Any = None,
        llm_getter: Any = None,
        tools_getter: Any = None,
        **_: Any,
    ) -> None:
        """Run deferred discovery, pick up kit-level model_resolver, discover siblings."""
        # --- Ordering check: AgentsExtension must be inner to HistoryExtension ---
        # when the configured output strategy tags subagent messages as
        # hidden-from-LLM. Our wrap_model filter must run inside History's
        # wrap_model so the ``ReplaceMessages(kept + response)`` commit sees
        # the full, un-filtered state and persists the subagent trace.
        # Mirrors TeamExtension's own ordering check.
        if self._strategy_tags_hidden:
            from langchain_agentkit.extensions.history.extension import HistoryExtension

            my_index: int | None = None
            history_index: int | None = None
            for i, ext in enumerate(extensions):
                if ext is self:
                    my_index = i
                elif isinstance(ext, HistoryExtension) and history_index is None:
                    history_index = i
            if (
                my_index is not None
                and history_index is not None
                and my_index < history_index
            ):
                raise ValueError(
                    "AgentsExtension must be declared AFTER HistoryExtension when "
                    "output_mode='trace_hidden' (or any strategy that tags messages "
                    f"with '{self._metadata_prefix}_hidden_from_llm'). The filter "
                    "wrap_model hook must run inner to HistoryExtension so the "
                    "subagent trace is retained in persisted state. "
                    f"AgentsExtension is at index {my_index}, "
                    f"HistoryExtension at index {history_index}."
                )

        # --- Pick up kit-level model_resolver ---
        if model_resolver is not None and self._model_resolver is None:
            self._model_resolver = model_resolver

        # --- Wire kit-level parent LLM / tools getters ---
        # Config-based agents without an explicit ``model`` inherit the
        # parent LLM via ``parent_llm_getter``. Without this wiring,
        # ``_delegate_agent_config`` would trip over a None getter at
        # tool-call time.
        if llm_getter is not None and self._parent_llm_getter is None:
            self._parent_llm_getter = llm_getter
        if tools_getter is not None and self._parent_tools_getter is _default_tools_getter:
            self._parent_tools_getter = tools_getter

        # --- Deferred backend discovery ---
        if self._deferred_path is not None and self._backend is not None:
            from langchain_agentkit.extensions.agents.discovery import (
                discover_agents_from_backend,
            )

            defs = await discover_agents_from_backend(self._backend, self._deferred_path)
            proxies = [_AgentConfigProxy(d) for d in defs]
            if proxies:
                self._agents_by_name = validate_agent_list(proxies)
            else:
                self._agents_by_name = {}
            self._has_config_agents = bool(proxies)
            self._deferred_path = None
            # Rebuild tools with discovered agents
            self._tools = tuple(self._create_tools())

        # --- Discover SkillsExtension sibling for skill preloading ---
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

    async def wrap_model(
        self,
        *,
        state: Any,
        handler: Any,
        runtime: Any,
    ) -> Any:
        """Strip hidden-tagged subagent messages from the LLM request.

        Activated only when the configured output strategy tags messages
        with ``{metadata_prefix}_hidden_from_llm=True``. Filters the
        per-request message list before passing to the handler; graph
        state is not mutated.

        Must run inner to any ``wrap_model`` hook that commits state
        via ``ReplaceMessages`` (today: :class:`HistoryExtension`).
        ``setup()`` enforces this ordering and raises ``ValueError`` at
        kit-construction time if the user's extension list violates it.
        """
        if not self._strategy_tags_hidden:
            return await handler(state)

        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return await handler(state)

        from langchain_agentkit.extensions.agents.filter import strip_hidden_from_llm

        filtered = strip_hidden_from_llm(messages, metadata_prefix=self._metadata_prefix)
        if len(filtered) == len(messages):
            return await handler(state)
        return await handler({**state, "messages": filtered})

    def _create_tools(self) -> list[BaseTool]:
        from langchain_agentkit.extensions.agents.tools import create_agent_tools

        needs_llm = self._ephemeral or self._has_config_agents
        return create_agent_tools(
            agents_by_name=self._agents_by_name,
            compiled_cache=self._compiled_cache,
            delegation_timeout=self._delegation_timeout,
            parent_tools_getter=lambda: self._parent_tools_getter(),
            ephemeral=self._ephemeral,
            parent_llm_getter=(lambda: self._parent_llm_getter()) if needs_llm else None,
            model_resolver=(
                lambda name: self._model_resolver(name) if self._model_resolver else None
            ),
            skills_resolver=(
                lambda names: self._skills_resolver(names) if self._skills_resolver else None  # type: ignore[arg-type, return-value]
            ),
            output_strategy=self._output_strategy,
            strategy_context=self._strategy_context,
        )

    def set_parent_tools_getter(self, getter: Any) -> None:
        self._parent_tools_getter = getter

    def set_parent_llm_getter(self, getter: Any) -> None:
        self._parent_llm_getter = getter

    @property
    def metadata_prefix(self) -> str:
        """Prefix used on ``response_metadata`` keys for subagent tags.

        Exposed so a sibling :class:`HideSubagentTraceExtension` can be
        configured from the same source of truth.
        """
        return self._metadata_prefix

    @property
    def agents_by_name(self) -> dict[str, Any]:
        return self._agents_by_name

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools  # type: ignore[return-value]

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        roster_lines = []
        for agent_name, agent_obj in self._agents_by_name.items():
            desc = getattr(agent_obj, "description", "") or "No description"
            tools_desc = _get_tools_description(agent_obj)
            roster_lines.append(f"- **{agent_name}**: {desc} (Tools: {tools_desc})")
        roster = "\n".join(roster_lines)
        dynamic_section = _DYNAMIC_SECTION if self._ephemeral else ""
        result = _agent_delegation_template.format(
            agent_roster=roster,
            dynamic_section=dynamic_section,
        )
        result += _PARALLEL_NOTE
        if self._default_conciseness:
            result += _CONCISENESS_DIRECTIVE
        return result
