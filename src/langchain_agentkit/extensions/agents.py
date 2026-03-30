"""AgentExtension — blocking subagent delegation with parallel support.

Two input modes:

1. **Programmatic** — pass executable agent objects::

    mw = AgentExtension(agents=[researcher, coder])

2. **Directory discovery** — pass a directory path to scan for agent .md files::

    mw = AgentExtension(agents="./agents")

    # Or with a custom backend (e.g. Daytona sandbox):
    mw = AgentExtension(agents="/agents", backend=my_backend)

Filesystem-discovered agents are "named dynamic agents" — they have a name,
description, and system prompt from the markdown body. At delegation time,
they use the parent's LLM (same mechanism as ephemeral agents).

Parallel delegation: the LLM calls Agent multiple times in one turn.
LangGraph's ToolNode executes them concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.frontmatter import parse_frontmatter, parse_frontmatter_string
from langchain_agentkit.validate import validate_name

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backend import BackendProtocol

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_agent_delegation_template = PromptTemplate.from_file(_PROMPTS_DIR / "agent_delegation.md")

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


def _strip_line_numbers(formatted: str) -> str:
    """Strip line-number prefixes from BackendProtocol.read() output.

    BackendProtocol.read() returns ``{line_num}\\t{content}`` per line.
    This strips the prefix to recover raw file content.
    """
    lines = []
    for line in formatted.splitlines(keepends=True):
        _, _, content = line.partition("\t")
        lines.append(content)
    return "".join(lines)


def _parse_comma_list(value: Any) -> list[str] | None:
    """Parse a comma-separated frontmatter value into a list of strings.

    Returns None if the value is absent or empty.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class AgentConfig:
    """Agent definition — used by both programmatic and file-based agents.

    At delegation time, these fields are resolved:
    - ``model``: resolved via ``model_resolver`` if string, or used as-is
    - ``tools``: filtered from parent's available tools by name
    - ``skills``: resolved by name → content concatenated into prompt
    - ``max_turns``: used as recursion limit on the compiled graph
    """

    name: str
    description: str
    prompt: str
    tools: list[str] | None = None
    model: str | None = None
    max_turns: int | None = None
    skills: list[str] | None = None



def _agent_config_from_metadata(
    metadata: dict[str, Any],
    content: str,
) -> AgentConfig | None:
    """Parse an AgentConfig from frontmatter metadata and body content.

    Returns None if the name is missing or invalid.
    """
    name = metadata.get("name", "")
    if not name or validate_name(name) is not None:
        return None

    max_turns_raw = metadata.get("maxTurns")
    max_turns = int(max_turns_raw) if max_turns_raw is not None else None

    return AgentConfig(
        name=name,
        description=metadata.get("description", ""),
        prompt=content,
        tools=_parse_comma_list(metadata.get("tools")),
        model=metadata.get("model") or None,
        max_turns=max_turns,
        skills=_parse_comma_list(metadata.get("skills")),
    )


def _discover_agents_from_directory(
    path: Path,
) -> list[AgentConfig]:
    """Discover agents by scanning a local directory for .md files."""
    if not path.is_dir():
        return []

    agents: list[AgentConfig] = []
    seen_names: set[str] = set()

    for md_file in sorted(path.rglob("*.md")):
        try:
            result = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            continue
        agent_config = _agent_config_from_metadata(result.metadata, result.content)
        if agent_config is None or agent_config.name in seen_names:
            continue
        seen_names.add(agent_config.name)
        agents.append(agent_config)

    return agents


def _discover_agents_from_backend(
    backend: BackendProtocol,
    path: str,
) -> list[AgentConfig]:
    """Discover agents via a BackendProtocol by globbing for .md files."""
    matches = backend.glob("**/*.md", path=path)
    agents: list[AgentConfig] = []
    seen_names: set[str] = set()

    for match in sorted(matches):
        try:
            formatted = backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            continue
        content = _strip_line_numbers(formatted)
        result = parse_frontmatter_string(content)
        agent_config = _agent_config_from_metadata(result.metadata, result.content)
        if agent_config is None or agent_config.name in seen_names:
            continue
        seen_names.add(agent_config.name)
        agents.append(agent_config)

    return agents


class _AgentConfigProxy:
    """Proxy that makes an AgentConfig look like an agent to the roster.

    Has ``name``, ``description``, and ``tools_inherit = False``.
    The delegation tool detects ``_agent_config`` attribute and
    routes to the definition-based delegation path.
    """

    def __init__(self, definition: AgentConfig) -> None:
        self.name = definition.name
        self.description = definition.description
        self.tools_inherit = False
        self._agent_config = definition


def _wrap_agents(agents: list[Any]) -> list[Any]:
    """Wrap AgentConfig instances in proxies, pass everything else through.

    Allows mixed lists of compiled StateGraphs, AgentLike objects, and
    AgentConfig definitions in a single agents list.
    """
    result = []
    for a in agents:
        if isinstance(a, AgentConfig):
            result.append(_AgentConfigProxy(a))
        else:
            result.append(a)
    return result


class AgentExtension(Extension):
    """Extension providing blocking subagent delegation via the Agent tool.

    Two input modes:

    - **List**: Pass agent objects (StateGraph, AgentLike, or AgentConfig).
      AgentConfig definitions are compiled at delegation time using the
      parent's LLM and resolved tools/skills.
    - **Path**: Pass a string or Path to a directory to scan for ``.md`` files
      with frontmatter. When ``backend`` is provided, discovery uses the
      backend's filesystem. Otherwise, scans the local OS filesystem.

    Example::

        ext = AgentExtension(agents=[
            researcher_graph,                          # compiled StateGraph
            AgentConfig(name="coder", prompt="..."),   # definition
        ])

    Args:
        agents: List of agent objects, or a string/Path to a directory.
        backend: Optional BackendProtocol for remote filesystem discovery.
        ephemeral: Enable dynamic (on-the-fly) agents in the Agent tool schema.
        default_conciseness: Append conciseness directive to delegation prompt.
        delegation_timeout: Max seconds to wait for a subagent response.

    Raises:
        ValueError: If ``agents`` list is empty or contains duplicates.
    """

    def __init__(
        self,
        agents: list[Any] | str | Path,
        backend: BackendProtocol | None = None,
        ephemeral: bool = False,
        default_conciseness: bool = True,
        delegation_timeout: float = 300.0,
    ) -> None:
        from langchain_agentkit.extensions import validate_agent_list

        if isinstance(agents, list):
            wrapped = _wrap_agents(agents)
            self._agents_by_name: dict[str, Any] = validate_agent_list(wrapped)
            self._has_config_agents = any(
                isinstance(a, _AgentConfigProxy) for a in wrapped
            )
        elif isinstance(agents, (str, Path)):
            if backend is not None:
                defs = _discover_agents_from_backend(backend, str(agents))
            else:
                defs = _discover_agents_from_directory(Path(agents))
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

        # Placeholder — resolved lazily by tools when parent context is available
        self._parent_tools_getter: Any = list
        self._parent_llm_getter: Any = None
        self._model_resolver: Any = None
        self._skills_resolver: Any = None

        self._tools = tuple(self._create_tools())

    def _create_tools(self) -> list[BaseTool]:
        """Create the unified Agent tool with closures over extensions state."""
        from langchain_agentkit.tools.agent import create_agent_tools

        # Filesystem/def agents always need parent LLM for delegation
        needs_llm = self._ephemeral or self._has_config_agents

        return create_agent_tools(
            agents_by_name=self._agents_by_name,
            compiled_cache=self._compiled_cache,
            delegation_timeout=self._delegation_timeout,
            parent_tools_getter=lambda: self._parent_tools_getter(),
            ephemeral=self._ephemeral,
            parent_llm_getter=(lambda: self._parent_llm_getter()) if needs_llm else None,
            model_resolver=(
                lambda name: self._model_resolver(name)
                if self._model_resolver else None
            ),
            skills_resolver=(
                lambda names: self._skills_resolver(names)
                if self._skills_resolver else None
            ),
        )

    def set_parent_tools_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's tools."""
        self._parent_tools_getter = getter

    def set_parent_llm_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's LLM."""
        self._parent_llm_getter = getter

    def set_model_resolver(self, resolver: Any) -> None:
        """Set the callable that resolves model name strings to BaseChatModel."""
        self._model_resolver = resolver

    def set_skills_resolver(self, resolver: Any) -> None:
        """Set the callable that resolves skill names to concatenated content."""
        self._skills_resolver = resolver

    @property
    def tools(self) -> list[BaseTool]:
        """The Agent tool provided by this extension."""
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Build the delegation prompt with agent roster."""
        template = _agent_delegation_template

        roster_lines = []
        for agent_name, agent_obj in self._agents_by_name.items():
            desc = getattr(agent_obj, "description", "") or "No description"
            roster_lines.append(f"- **{agent_name}**: {desc}")

        roster = "\n".join(roster_lines)
        dynamic_section = _DYNAMIC_SECTION if self._ephemeral else ""
        result = template.format(agent_roster=roster, dynamic_section=dynamic_section)

        if self._default_conciseness:
            result += _CONCISENESS_DIRECTIVE

        return result
