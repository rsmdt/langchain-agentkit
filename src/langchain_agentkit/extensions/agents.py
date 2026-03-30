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


@dataclass(frozen=True)
class FilesystemAgentDef:
    """Agent definition discovered from a markdown file.

    These are compiled at delegation time using the parent's LLM,
    following the same path as ephemeral/dynamic agents.
    """

    name: str
    description: str
    instructions: str


def _discover_agents_from_directory(
    path: Path,
) -> list[FilesystemAgentDef]:
    """Discover agents by scanning a local directory for .md files."""
    if not path.is_dir():
        return []

    agents: list[FilesystemAgentDef] = []
    seen_names: set[str] = set()

    for md_file in sorted(path.rglob("*.md")):
        try:
            result = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            continue
        name = result.metadata.get("name", "")
        if not name or validate_name(name) is not None:
            continue
        description = result.metadata.get("description", "")

        if name in seen_names:
            continue
        seen_names.add(name)
        agents.append(FilesystemAgentDef(
            name=name,
            description=description,
            instructions=result.content,
        ))

    return agents


def _discover_agents_from_backend(
    backend: BackendProtocol,
    path: str,
) -> list[FilesystemAgentDef]:
    """Discover agents via a BackendProtocol by globbing for .md files."""
    matches = backend.glob("**/*.md", path=path)
    agents: list[FilesystemAgentDef] = []
    seen_names: set[str] = set()

    for match in sorted(matches):
        try:
            formatted = backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            continue
        content = _strip_line_numbers(formatted)
        result = parse_frontmatter_string(content)
        name = result.metadata.get("name", "")
        if not name or validate_name(name) is not None:
            continue
        description = result.metadata.get("description", "")

        if name in seen_names:
            continue
        seen_names.add(name)
        agents.append(FilesystemAgentDef(
            name=name,
            description=description,
            instructions=result.content,
        ))

    return agents


class _FilesystemAgentProxy:
    """Proxy that makes a FilesystemAgentDef look like an agent to the roster.

    Has ``name``, ``description``, and ``tools_inherit = False``.
    The delegation tool detects ``_filesystem_agent_def`` attribute and
    routes to the dynamic agent path with the stored prompt.
    """

    def __init__(self, definition: FilesystemAgentDef) -> None:
        self.name = definition.name
        self.description = definition.description
        self.tools_inherit = False
        self._filesystem_agent_def = definition


class AgentExtension(Extension):
    """Extension providing blocking subagent delegation via the Agent tool.

    Two input modes:

    - **List**: Pass agent objects (StateGraph/AgentLike) directly.
    - **Path**: Pass a string or Path to a directory to scan for ``.md`` files
      with frontmatter. When ``backend`` is provided, discovery uses the
      backend's filesystem. Otherwise, scans the local OS filesystem.

    Filesystem-discovered agents use the parent's LLM at delegation time.

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
            self._agents_by_name: dict[str, Any] = validate_agent_list(agents)
            self._has_filesystem_agents = False
        elif isinstance(agents, (str, Path)):
            if backend is not None:
                defs = _discover_agents_from_backend(backend, str(agents))
            else:
                defs = _discover_agents_from_directory(Path(agents))
            proxies = [_FilesystemAgentProxy(d) for d in defs]
            if not proxies:
                self._agents_by_name = {}
            else:
                self._agents_by_name = validate_agent_list(proxies)
            self._has_filesystem_agents = True
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

        self._tools = tuple(self._create_tools())

    def _create_tools(self) -> list[BaseTool]:
        """Create the unified Agent tool with closures over extensions state."""
        from langchain_agentkit.tools.agent import create_agent_tools

        # Filesystem agents always need parent LLM for delegation
        needs_llm = self._ephemeral or self._has_filesystem_agents

        return create_agent_tools(
            agents_by_name=self._agents_by_name,
            compiled_cache=self._compiled_cache,
            delegation_timeout=self._delegation_timeout,
            parent_tools_getter=lambda: self._parent_tools_getter(),
            ephemeral=self._ephemeral,
            parent_llm_getter=(lambda: self._parent_llm_getter()) if needs_llm else None,
        )

    def set_parent_tools_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's tools."""
        self._parent_tools_getter = getter

    def set_parent_llm_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's LLM."""
        self._parent_llm_getter = getter

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
