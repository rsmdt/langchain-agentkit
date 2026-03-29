"""AgentKit — composition engine for Extension instances.

``AgentKit`` merges tools and prompts from an ordered list of extensions into
a single, unified surface that any LangGraph node can consume.

Use ``AgentKit`` when you need full control over graph topology — multi-node
graphs, a shared ``ToolNode``, custom routing, or any setup where the
higher-level ``agent`` metaclass is too opinionated.

Example::

    kit = AgentKit(extensions=[
        SkillsExtension("skills/"),
        TasksExtension(),
    ])

    # In any graph node:
    all_tools = my_tools + kit.tools
    system_prompt = my_template + "\\n\\n" + kit.prompt(state, runtime)

Prompt templates can be loaded from files or provided as inline strings::

    kit = AgentKit(extensions=exts, prompt="You are a helpful assistant.")
    kit = AgentKit(extensions=exts, prompt=Path("prompts/system.txt"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.extension import Extension


class AgentKit:
    """Composes extensions into unified tools + prompt.

    Use directly when you need full control over graph topology
    (multi-node graphs, shared ToolNode, custom routing).

    Example::

        kit = AgentKit(extensions=[
            SkillsExtension("skills/"),
            TasksExtension(),
        ])

        # In any graph node:
        all_tools = my_tools + kit.tools
        prompt = my_template + "\\n\\n" + kit.prompt(state, runtime)
    """

    def __init__(
        self,
        extensions: list[Extension] | None = None,
        prompt: str | Path | list[str | Path] | None = None,
    ) -> None:
        self._extensions = self._resolve_dependencies(list(extensions or []))
        self._prompt = _load_prompt(prompt)
        self._tools_cache: list[BaseTool] | None = None

    @staticmethod
    def _resolve_dependencies(extensions: list) -> list:
        """Resolve extension dependencies. Auto-add missing dependencies.

        Uses isinstance/type for identity. Dependencies declared via
        the optional dependencies() method are added if not already present.
        """
        resolved = list(extensions)
        seen_types = {type(ext) for ext in resolved}

        # Iterate a copy — resolved may grow during iteration
        for ext in list(resolved):
            deps_fn = getattr(ext, "dependencies", None)
            if not callable(deps_fn):
                continue
            for dep in deps_fn():
                if type(dep) not in seen_types:
                    resolved.append(dep)
                    seen_types.add(type(dep))

        return resolved

    @property
    def tools(self) -> list[BaseTool]:
        """All tools from all extensions, deduplicated by name.

        Collected once on first access, then cached. First extension wins
        on name collisions.
        """
        if self._tools_cache is None:
            seen: set[str] = set()
            tools: list[BaseTool] = []
            for ext in self._extensions:
                for tool in ext.tools:
                    if tool.name not in seen:
                        seen.add(tool.name)
                        tools.append(tool)
            self._tools_cache = tools
        return self._tools_cache

    @property
    def state_schema(self) -> type:
        """Compose state schema from ``AgentKitState`` + all extension schemas.

        Each extension may declare a ``state_schema`` property returning a
        TypedDict mixin. These are combined via multiple inheritance into a
        single composed type. Extensions without ``state_schema`` (or returning
        ``None``) are skipped.
        """
        from langchain_agentkit.state import AgentKitState

        bases: list[type] = [AgentKitState]
        seen: set[int] = {id(AgentKitState)}
        for ext in self._extensions:
            schema = getattr(ext, "state_schema", None)
            if schema is not None and id(schema) not in seen:
                seen.add(id(schema))
                bases.append(schema)
        if len(bases) == 1:
            return AgentKitState
        return type("ComposedState", tuple(bases), {})

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Compose prompt from template + all extension sections.

        Called on every LLM invocation. Each extension contributes
        a section in stack order. Sections joined with double newline.
        """
        sections = [self._prompt] if self._prompt else []
        for ext in self._extensions:
            section = ext.prompt(state, runtime)
            if section:
                sections.append(section)
        return "\n\n".join(sections)


def _load_prompt_source(source: str | Path) -> str:
    """Load a single prompt source from file path or return inline string."""
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text()
    return str(source)


def _load_prompt(source: str | Path | list[str | Path] | None) -> str:
    """Load and concatenate prompt sources."""
    if source is None:
        return ""
    if isinstance(source, (str, Path)):
        return _load_prompt_source(source)
    parts = [_load_prompt_source(s) for s in source]
    return "\n\n".join(p for p in parts if p)
