"""AgentExtension — blocking subagent delegation with parallel support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.agents.discovery import (
    discover_agents_from_backend,
    discover_agents_from_directory,
)
from langchain_agentkit.extensions.agents.types import (
    _AgentConfigProxy,
    _wrap_agents,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import BackendProtocol

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_agent_delegation_template = PromptTemplate.from_file(_PROMPT_FILE)

_CONCISENESS_DIRECTIVE = (
    "\n\nWhen reporting delegation results, be concise. "
    "Synthesize the key findings — don't repeat the subagent's full response verbatim."
)

_DYNAMIC_SECTION = """\
**To a custom agent** — define its role with a system prompt:
```
Agent(agent={prompt: "You are a legal expert..."}, message="...")
```
Custom agents are reasoning-only — they cannot use tools."""


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


def _validate_agent_list(agents: list[Any]) -> dict[str, Any]:
    """Validate agent list and return agents_by_name dict."""
    if not agents:
        raise ValueError("agents list cannot be empty")
    names = [getattr(g, "name", None) for g in agents]
    if any(n is None for n in names):
        raise ValueError("All agents must have a name")
    if len(set(names)) != len(names):
        dupes = [n for n in names if names.count(n) > 1]
        raise ValueError(f"Duplicate agent names: {set(dupes)}")
    return {name: agent for name, agent in zip(names, agents, strict=True)}  # type: ignore[misc]


def _resolve_agent(agent_name: str, agents_by_name: dict[str, Any]) -> Any:
    """Look up an agent by name."""
    import logging

    from langchain_core.tools import ToolException

    if agent_name not in agents_by_name:
        logging.getLogger(__name__).warning(
            "Agent '%s' not found. Registered: %s",
            agent_name,
            sorted(agents_by_name.keys()),
        )
        raise ToolException(
            f"Agent '{agent_name}' not found. Check the agent roster for available names."
        )
    return agents_by_name[agent_name]


class AgentExtension(Extension):
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
    """

    def __init__(
        self,
        *,
        agents: list[Any] | str | Path,
        backend: BackendProtocol | None = None,
        ephemeral: bool = False,
        default_conciseness: bool = True,
        delegation_timeout: float = 300.0,
    ) -> None:
        if isinstance(agents, list):
            wrapped = _wrap_agents(agents)
            self._agents_by_name: dict[str, Any] = _validate_agent_list(wrapped)
            self._has_config_agents = any(isinstance(a, _AgentConfigProxy) for a in wrapped)
        elif isinstance(agents, (str, Path)):
            if backend is not None:
                defs = discover_agents_from_backend(backend, str(agents))
            else:
                defs = discover_agents_from_directory(Path(agents))
            proxies = [_AgentConfigProxy(d) for d in defs]
            if proxies:
                self._agents_by_name = _validate_agent_list(proxies)
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

        self._parent_tools_getter: Any = list
        self._parent_llm_getter: Any = None
        self._model_resolver: Any = None
        self._skills_resolver: Any = None

        self._tools = tuple(self._create_tools())

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
            resolve_agent_fn=_resolve_agent,
        )

    def set_parent_tools_getter(self, getter: Any) -> None:
        self._parent_tools_getter = getter

    def set_parent_llm_getter(self, getter: Any) -> None:
        self._parent_llm_getter = getter

    def set_model_resolver(self, resolver: Any) -> None:
        self._model_resolver = resolver

    def set_skills_resolver(self, resolver: Any) -> None:
        self._skills_resolver = resolver

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
        if self._default_conciseness:
            result += _CONCISENESS_DIRECTIVE
        return result
