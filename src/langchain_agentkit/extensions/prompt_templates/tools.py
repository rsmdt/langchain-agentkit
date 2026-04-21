"""``RunCommand`` tool — invoke a discovered template by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.prompt_templates.types import PromptTemplate

_RUN_COMMAND_DESCRIPTION = """Expand and return a named prompt template.

Templates live in ``.agentkit/commands/`` (or are registered directly).
The result is the rendered template body — the model should treat it as
the next instruction from the user.

Arguments:
- ``name`` — the template identifier (see roster in the system prompt).
- ``args`` — raw argument string using bash-like quoting (``"with space"``,
  ``'single too'``, ``\\\\\"escaped\\\\\"``). Leave empty for templates that
  take no arguments.

Reserved for predefined workflows. Prefer writing a new template over
constructing ad-hoc multi-step instructions."""


class _RunCommandInput(BaseModel):
    name: str = Field(description="Template name — see the system prompt roster.")
    args: str = Field(default="", description="Raw argument string, bash-like quoting.")


def build_run_command_tool(
    templates: dict[str, PromptTemplate],
) -> BaseTool:
    """Build the ``RunCommand`` tool backed by ``templates``."""
    from langchain_agentkit.extensions.prompt_templates.parser import parse_args
    from langchain_agentkit.extensions.prompt_templates.render import expand_template

    async def _run(name: str, args: str = "") -> str:
        template = templates.get(name)
        if template is None:
            known = ", ".join(sorted(templates)) or "(none)"
            return f"Unknown template {name!r}. Known templates: {known}"
        try:
            parsed = parse_args(args)
        except ValueError as exc:
            return f"Invalid args for {name!r}: {exc}"
        return expand_template(template.body, parsed.positional)

    return StructuredTool.from_function(
        coroutine=_run,
        name="RunCommand",
        description=_RUN_COMMAND_DESCRIPTION,
        args_schema=_RunCommandInput,
    )


def render_template_roster(templates: dict[str, PromptTemplate]) -> str:
    """Render the template list for inclusion in the system prompt."""
    if not templates:
        return "(No commands registered.)"
    lines: list[str] = []
    for name in sorted(templates):
        t = templates[name]
        hint = f" {t.argument_hint}" if t.argument_hint else ""
        lines.append(f"- `{name}`{hint} — {t.description}")
    return "\n".join(lines)
