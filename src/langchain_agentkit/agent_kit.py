"""AgentKit — composition engine for Middleware instances.

``AgentKit`` merges tools and prompts from an ordered list of middleware into
a single, unified surface that any LangGraph node can consume.

Use ``AgentKit`` when you need full control over graph topology — multi-node
graphs, a shared ``ToolNode``, custom routing, or any setup where the
higher-level ``node`` metaclass is too opinionated.

Example::

    kit = AgentKit([
        SkillsMiddleware("skills/"),
        TasksMiddleware(),
    ])

    # In any graph node:
    all_tools = my_tools + kit.tools
    system_prompt = my_template + "\\n\\n" + kit.prompt(state, config)

Prompt templates can be loaded from files or provided as inline strings::

    kit = AgentKit(middleware, prompt="You are a helpful assistant.")
    kit = AgentKit(middleware, prompt=Path("prompts/system.txt"))
    kit = AgentKit(middleware, prompt=["prompts/base.txt", "prompts/persona.txt"])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

    from langchain_agentkit.middleware import Middleware


class AgentKit:
    """Composes middleware into unified tools + prompt.

    Use directly when you need full control over graph topology
    (multi-node graphs, shared ToolNode, custom routing).

    Example::

        kit = AgentKit([
            SkillsMiddleware("skills/"),
            TasksMiddleware(),
        ])

        # In any graph node:
        all_tools = my_tools + kit.tools
        prompt = my_template + "\\n\\n" + kit.prompt(state, config)
    """

    def __init__(
        self,
        middleware: list[Middleware],
        prompt: str | Path | list[str | Path] | None = None,
    ) -> None:
        self._middleware = list(middleware)
        self._template = _load_templates(prompt)
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """All tools from all middleware, deduplicated by name.

        Collected once on first access, then cached. First middleware wins
        on name collisions.
        """
        if self._tools_cache is None:
            seen: set[str] = set()
            tools: list[BaseTool] = []
            for mw in self._middleware:
                for tool in mw.tools:
                    if tool.name not in seen:
                        seen.add(tool.name)
                        tools.append(tool)
            self._tools_cache = tools
        return self._tools_cache

    def prompt(self, state: dict, config: RunnableConfig) -> str:
        """Compose prompt from template + all middleware sections.

        Called on every LLM invocation. Each middleware contributes
        a section in stack order. Sections joined with double newline.
        """
        sections = [self._template] if self._template else []
        for mw in self._middleware:
            section = mw.prompt(state, config)
            if section:
                sections.append(section)
        return "\n\n".join(sections)


def _load_template(source: str | Path) -> str:
    """Load a single prompt template from file path or return inline string."""
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text()
    return str(source)


def _load_templates(source: str | Path | list[str | Path] | None) -> str:
    """Load and concatenate prompt templates.

    Accepts a single template or a list. Templates are loaded from file
    paths or treated as inline strings, then joined with double newline.
    """
    if source is None:
        return ""
    if isinstance(source, (str, Path)):
        return _load_template(source)
    # List of templates
    parts = [_load_template(s) for s in source]
    return "\n\n".join(p for p in parts if p)
