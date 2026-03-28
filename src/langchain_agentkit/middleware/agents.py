"""AgentMiddleware — blocking subagent delegation with parallel support.

Usage::

    from langchain_agentkit import agent, AgentMiddleware

    class researcher(agent):
        llm = ChatOpenAI(model="gpt-4o-mini")
        description = "Research specialist"
        tools = [web_search]
        async def handler(state, *, llm, tools, prompt): ...

    class lead(agent):
        llm = ChatOpenAI(model="gpt-4o")
        middleware = [AgentMiddleware([researcher])]
        async def handler(state, *, llm, tools, prompt): ...

Parallel delegation: the LLM calls Agent multiple times in one turn.
LangGraph's ToolNode executes them concurrently.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

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


class AgentMiddleware:
    """Middleware providing blocking subagent delegation via the Agent tool.

    Parallel delegation: LLM calls multiple Agent tools in one turn.
    LangGraph's ToolNode executes them concurrently.

    Args:
        agents: List of StateGraph objects created via the ``agent`` metaclass.
            Each must have ``agentkit_name`` and ``agentkit_description`` attrs.
        ephemeral: Enable dynamic (on-the-fly) agents in the Agent tool schema.
        default_conciseness: Append conciseness directive to delegation prompt.
        delegation_timeout: Max seconds to wait for a subagent response.

    Example::

        mw = AgentMiddleware(
            [researcher, coder],
            ephemeral=True,
            delegation_timeout=120.0,
        )
    """

    def __init__(
        self,
        agents: list[Any],
        ephemeral: bool = False,
        default_conciseness: bool = True,
        delegation_timeout: float = 300.0,
    ) -> None:
        from langchain_agentkit.middleware import validate_agent_list

        self._agents_by_name: dict[str, Any] = validate_agent_list(agents)
        self._ephemeral = ephemeral
        self._default_conciseness = default_conciseness
        self._delegation_timeout = delegation_timeout
        self._compiled_cache: dict[str, Any] = {}

        # Placeholder — resolved lazily by tools when parent context is available
        self._parent_tools_getter: Any = list
        self._parent_llm_getter: Any = None

        self._tools = tuple(self._create_tools())

    def _create_tools(self) -> list[BaseTool]:
        """Create the unified Agent tool with closures over middleware state."""
        from langchain_agentkit.tools.agent import create_agent_tools

        return create_agent_tools(
            agents_by_name=self._agents_by_name,
            compiled_cache=self._compiled_cache,
            delegation_timeout=self._delegation_timeout,
            parent_tools_getter=lambda: self._parent_tools_getter(),
            ephemeral=self._ephemeral,
            parent_llm_getter=(lambda: self._parent_llm_getter()) if self._ephemeral else None,
        )

    def set_parent_tools_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's tools.

        Called by the framework when wiring up the agent graph,
        enabling ``tools="inherit"`` resolution at delegation time.
        """
        self._parent_tools_getter = getter

    def set_parent_llm_getter(self, getter: Any) -> None:
        """Set the callable that returns the parent agent's LLM.

        Called by the framework when wiring up the agent graph,
        enabling dynamic agents to use the parent's LLM.
        """
        self._parent_llm_getter = getter

    @property
    def tools(self) -> list[BaseTool]:
        """The Agent tool provided by this middleware."""
        return self._tools

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Build the delegation prompt with agent roster."""
        template = _agent_delegation_template

        roster_lines = []
        for agent_graph in self._agents_by_name.values():
            name = getattr(agent_graph, "agentkit_name", "unknown")
            description = getattr(agent_graph, "agentkit_description", "") or "No description"
            roster_lines.append(f"- **{name}**: {description}")

        roster = "\n".join(roster_lines)
        dynamic_section = _DYNAMIC_SECTION if self._ephemeral else ""
        result = template.format(agent_roster=roster, dynamic_section=dynamic_section)

        if self._default_conciseness:
            result += _CONCISENESS_DIRECTIVE

        return result
