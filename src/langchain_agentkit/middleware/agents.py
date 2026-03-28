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

Parallel delegation: the LLM calls Delegate multiple times in one turn.
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

_CONCISENESS_DIRECTIVE = (
    "\n\nWhen reporting delegation results, be concise. "
    "Synthesize the key findings — don't repeat the subagent's full response verbatim."
)


class AgentMiddleware:
    """Middleware providing blocking subagent delegation.

    Parallel delegation: LLM calls multiple Delegate tools in one turn.
    LangGraph's ToolNode executes them concurrently.

    Args:
        agents: List of StateGraph objects created via the ``agent`` metaclass.
            Each must have ``agentkit_name`` and ``agentkit_description`` attrs.
        ephemeral: Enable the DelegateEphemeral tool for ad-hoc reasoning agents.
        scoped_context: Subagent receives only the task message, not full history.
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
        scoped_context: bool = True,
        default_conciseness: bool = True,
        delegation_timeout: float = 300.0,
    ) -> None:
        if not agents:
            raise ValueError("agents list cannot be empty")

        names = [getattr(g, "agentkit_name", None) for g in agents]
        if any(n is None for n in names):
            raise ValueError(
                "All agents must have agentkit_name (use the agent metaclass)"
            )

        if len(set(names)) != len(names):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate agent names: {set(dupes)}")

        self._agents_by_name: dict[str, Any] = {
            getattr(g, "agentkit_name"): g for g in agents
        }
        self._agents = list(agents)
        self._ephemeral = ephemeral
        self._scoped_context = scoped_context
        self._default_conciseness = default_conciseness
        self._delegation_timeout = delegation_timeout
        self._compiled_cache: dict[str, Any] = {}

        # Placeholder — resolved lazily by tools when parent context is available
        self._parent_tools_getter: Any = None
        self._parent_llm_getter: Any = None

        self._tools = self._create_tools()

    def _create_tools(self) -> list[BaseTool]:
        """Create delegation tools with closures over middleware state."""
        from langchain_agentkit.tools.agent import create_agent_tools

        return create_agent_tools(
            agents_by_name=self._agents_by_name,
            compiled_cache=self._compiled_cache,
            delegation_timeout=self._delegation_timeout,
            parent_tools_getter=self._get_parent_tools,
            ephemeral=self._ephemeral,
            parent_llm_getter=self._get_parent_llm if self._ephemeral else None,
        )

    def _get_parent_tools(self) -> list[BaseTool]:
        """Lazy getter for parent tools — resolved at delegation time."""
        if self._parent_tools_getter is not None:
            return self._parent_tools_getter()
        return []

    def _get_parent_llm(self) -> Any:
        """Lazy getter for parent LLM — resolved at delegation time."""
        if self._parent_llm_getter is not None:
            return self._parent_llm_getter()
        raise ValueError(
            "Parent LLM not available. Ensure the middleware is attached to an agent."
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
        enabling DelegateEphemeral to use the parent's LLM.
        """
        self._parent_llm_getter = getter

    @property
    def tools(self) -> list[BaseTool]:
        """Delegation tools provided by this middleware."""
        return list(self._tools)

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Build the delegation prompt with agent roster."""
        template = PromptTemplate.from_file(_PROMPTS_DIR / "agent_delegation.md")

        roster_lines = []
        for agent_graph in self._agents:
            name = getattr(agent_graph, "agentkit_name", "unknown")
            description = getattr(agent_graph, "agentkit_description", "") or "No description"
            roster_lines.append(f"- **{name}**: {description}")

        roster = "\n".join(roster_lines)
        result = template.format(agent_roster=roster)

        if self._default_conciseness:
            result += _CONCISENESS_DIRECTIVE

        return result

    @property
    def state_schema(self) -> type:
        """SubAgentState adds delegation_log to graph state."""
        from langchain_agentkit.state import SubAgentState

        return SubAgentState
