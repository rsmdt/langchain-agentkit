"""PromptTemplateExtension — exposes named, parameterized prompt commands.

Users drop markdown templates in ``.agentkit/commands/*.md`` with
frontmatter (``name``, ``description``, optional ``argument-hint``), or
pass :class:`PromptTemplate` instances directly. The extension registers
a ``RunCommand`` tool so the model can invoke templates by name, and
surfaces the roster in the system prompt so the model knows what's
available. Programmatic callers can also use
:meth:`PromptTemplateExtension.render` to expand a template directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.prompt_templates.discovery import (
    discover_templates_from_directory,
)
from langchain_agentkit.extensions.prompt_templates.parser import parse_args
from langchain_agentkit.extensions.prompt_templates.render import expand_template
from langchain_agentkit.extensions.prompt_templates.tools import (
    build_run_command_tool,
    render_template_roster,
)
from langchain_agentkit.extensions.prompt_templates.types import (
    PromptTemplate,
    PromptTemplateError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import BackendProtocol


class PromptTemplateExtension(Extension):
    """Register discoverable prompt templates as ``RunCommand`` targets.

    Args:
        templates: Either a list of :class:`PromptTemplate` instances OR
            a directory path (``str`` / ``Path``) to scan for ``*.md``
            files with frontmatter.
        backend: Optional :class:`BackendProtocol`. When provided with a
            path, discovery is deferred to :meth:`setup` (async).
    """

    def __init__(
        self,
        *,
        templates: Sequence[PromptTemplate] | str | Path,
        backend: BackendProtocol | None = None,
    ) -> None:
        self._backend = backend
        self._deferred_path: str | None = None
        if isinstance(templates, (list, tuple)):
            self._templates: dict[str, PromptTemplate] = {t.name: t for t in templates}
        elif isinstance(templates, (str, Path)):
            if backend is not None:
                self._deferred_path = str(templates)
                self._templates = {}
            else:
                discovered = discover_templates_from_directory(Path(templates))
                self._templates = {t.name: t for t in discovered}
        else:
            raise PromptTemplateError(
                f"templates must be a list of PromptTemplate, str, or Path; "
                f"got {type(templates).__name__}"
            )
        self._tools_cache: list[BaseTool] | None = None

    async def setup(self, **_: Any) -> None:  # type: ignore[override]
        if self._deferred_path is not None and self._backend is not None:
            from langchain_agentkit.extensions.prompt_templates.discovery import (
                discover_templates_from_backend,
            )

            discovered = await discover_templates_from_backend(self._backend, self._deferred_path)
            self._templates = {t.name: t for t in discovered}
            self._deferred_path = None
            self._tools_cache = None

    @property
    def templates(self) -> dict[str, PromptTemplate]:
        """Read-only view of the registered templates (copy)."""
        return dict(self._templates)

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            self._tools_cache = [build_run_command_tool(self._templates)]
        return self._tools_cache

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> dict[str, str]:
        roster = render_template_roster(self._templates)
        static = (
            "## Commands\n\n"
            "The `RunCommand` tool expands predefined workflow templates "
            "by name. Invoke it when one of the registered templates "
            "matches the task.\n\n"
            f"Registered commands:\n{roster}"
        )
        return {"prompt": static}

    def render(self, name: str, args: str | Sequence[str] = ()) -> str:
        """Expand a template directly — useful for harness-level invocation.

        Raises :class:`PromptTemplateError` when the template is unknown
        or the arguments cannot be parsed.
        """
        template = self._templates.get(name)
        if template is None:
            known = ", ".join(sorted(self._templates)) or "(none)"
            raise PromptTemplateError(f"Unknown template {name!r}. Known templates: {known}")
        parsed = parse_args(args)
        return expand_template(template.body, parsed.positional)
