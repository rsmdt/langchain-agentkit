"""AgentKit — composition engine for Extension instances.

``AgentKit`` merges tools and prompts from an ordered list of extensions into
a single, unified surface that any LangGraph node can consume.

Use ``AgentKit`` when you need full control over graph topology — multi-node
graphs, a shared ``ToolNode``, custom routing, or any setup where the
higher-level ``agent`` metaclass is too opinionated.

Example::

    kit = AgentKit(extensions=[
        SkillsExtension(skills="skills/"),
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
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.extension import Extension


class AgentKit:
    """Composes extensions into unified tools + prompt.

    Use directly when you need full control over graph topology
    (multi-node graphs, shared ToolNode, custom routing).

    Example::

        kit = AgentKit(extensions=[
            SkillsExtension(skills="skills/"),
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
        model_resolver: Callable[[str], BaseChatModel] | None = None,
    ) -> None:
        self._extensions = self._resolve_dependencies(list(extensions or []))
        self._prompt = _load_prompt(prompt)
        self._tools_cache: list[BaseTool] | None = None
        self._model_resolver = model_resolver
        self._wire_extensions()

    def _wire_extensions(self) -> None:  # noqa: C901
        """Wire cross-extension callbacks after all extensions are resolved.

        - Sets model_resolver and skills_resolver on AgentExtension if present.
        - Detects HITLExtension and notifies FilesystemExtension for permission gating.
        """
        from langchain_agentkit.extensions.agents import AgentExtension
        from langchain_agentkit.extensions.filesystem import FilesystemExtension
        from langchain_agentkit.extensions.hitl import HITLExtension
        from langchain_agentkit.extensions.skills import SkillsExtension
        from langchain_agentkit.extensions.tasks import TasksExtension
        from langchain_agentkit.extensions.teams import TeamExtension

        # Find sibling extensions for cross-wiring
        skills_ext = None
        has_hitl = False
        has_team = False
        for ext in self._extensions:
            if isinstance(ext, SkillsExtension):
                skills_ext = ext
            if isinstance(ext, HITLExtension):
                has_hitl = True
            if isinstance(ext, TeamExtension):
                has_team = True

        # Wire team_active flag into TasksExtension when teams are present
        if has_team:
            for ext in self._extensions:
                if isinstance(ext, TasksExtension) and not ext._team_active:
                    ext._team_active = True
                    # Rebuild tools with team-aware descriptions
                    from langchain_agentkit.extensions.tasks.tools import create_task_tools

                    ext._tools = tuple(create_task_tools(team_active=True))
                    self._tools_cache = None  # invalidate cached tools

        for ext in self._extensions:
            if isinstance(ext, AgentExtension):
                if self._model_resolver is not None:
                    ext.set_model_resolver(self._model_resolver)
                if skills_ext is not None:
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

                    ext.set_skills_resolver(_resolve_skills)

            # Wire HITL availability into FilesystemExtension for permission gating
            if isinstance(ext, FilesystemExtension) and has_hitl:
                ext.set_hitl_available(True)

    @staticmethod
    def _resolve_dependencies(extensions: list[Any]) -> list[Any]:
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

    @property
    def model_resolver(self) -> Callable[[str], BaseChatModel] | None:
        """The model resolver for string-based model references."""
        return self._model_resolver

    def resolve_model(self, model: Any) -> Any:
        """Resolve a model reference to a BaseChatModel instance.

        If *model* is a string, passes it through ``model_resolver``.
        Otherwise returns it as-is (assumed to be a BaseChatModel already).

        Raises:
            ValueError: If *model* is a string but no ``model_resolver`` is configured.
        """
        if isinstance(model, str):
            if self._model_resolver is None:
                raise ValueError(
                    f"model='{model}' is a string but no model_resolver is configured "
                    f"on AgentKit. Pass model_resolver=<callable> to AgentKit()."
                )
            return self._model_resolver(model)
        return model

    def _collect_contributions(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> tuple[list[str], list[str]]:
        """Single-pass collection of prompt sections and reminders.

        Iterates each extension's ``prompt()`` exactly once, splitting
        the result into system-prompt sections and ephemeral reminders.

        Returns:
            (prompt_parts, reminder_parts) — both are lists of non-empty strings.

        Extension ``prompt()`` can return:
        - ``str`` — goes into system prompt
        - ``dict`` with ``prompt`` and/or ``reminder`` keys — prompt goes
          to system prompt, reminder goes to ephemeral message
        - ``None`` — no contribution
        """
        prompt_parts: list[str] = [self._prompt] if self._prompt else []
        reminder_parts: list[str] = []
        for ext in self._extensions:
            result = ext.prompt(state, runtime)
            if result is None:
                continue
            if isinstance(result, str):
                if result:
                    prompt_parts.append(result)
            elif isinstance(result, dict):
                p = result.get("prompt", "")
                r = result.get("reminder", "")
                if p:
                    prompt_parts.append(p)
                if r:
                    reminder_parts.append(r)
        return prompt_parts, reminder_parts

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Compose system prompt from template + extension prompt sections.

        Called on every LLM invocation. Each extension contributes
        a section in stack order. Sections joined with double newline.

        Extensions returning a dict with ``prompt``/``reminder`` keys
        have only their ``prompt`` value collected here. The ``reminder``
        value is collected separately by :meth:`system_reminder`.
        """
        prompt_parts, _ = self._collect_contributions(state, runtime)
        return "\n\n".join(prompt_parts)

    def system_reminder(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Collect ephemeral reminder content from all extensions.

        Returns combined text that will be appended to the messages
        array as a ``HumanMessage`` wrapped in ``<system-reminder>``
        tags. This message is added at LLM call time and **never**
        stored in ``state["messages"]``.

        Only extensions returning a dict with a non-empty
        ``reminder`` value contribute here.
        """
        _, reminder_parts = self._collect_contributions(state, runtime)
        return "\n\n".join(reminder_parts) if reminder_parts else ""


def _load_prompt_source(source: str | Path) -> str:
    """Load a single prompt source from file path or return inline string."""
    text = str(source)
    # Skip path resolution for strings that are clearly inline prompts
    # (contain newlines or exceed OS filename limits).
    if "\n" in text or len(text) > 255:
        return text
    path = Path(text)
    try:
        if path.exists() and path.is_file():
            return path.read_text()
    except OSError:
        pass
    return text


def _load_prompt(source: str | Path | list[str | Path] | None) -> str:
    """Load and concatenate prompt sources."""
    if source is None:
        return ""
    if isinstance(source, (str, Path)):
        return _load_prompt_source(source)
    parts = [_load_prompt_source(s) for s in source]
    return "\n\n".join(p for p in parts if p)
