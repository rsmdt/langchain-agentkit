"""SkillsExtension — skill loading with two input modes.

Provides the ``Skill`` tool for progressive disclosure of skill instructions.

Two modes:

1. **Programmatic** — pass ``SkillConfig`` objects directly::

    ext = SkillsExtension(skills=[SkillConfig(name="research", ...)])

2. **Directory discovery** — pass a directory path to scan for SKILL.md files::

    ext = SkillsExtension(skills="./skills")
    ext = SkillsExtension(skills="/skills", backend=my_backend)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.skills.discovery import (
    discover_skills_from_directory,
)
from langchain_agentkit.extensions.skills.tools import build_skill_tool

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import BackendProtocol
    from langchain_agentkit.extensions.skills.types import SkillConfig

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_skills_system_prompt = PromptTemplate.from_file(_PROMPT_FILE)


class SkillsExtension(Extension):
    """Extension providing the Skill tool for progressive disclosure.

    Args:
        skills: Either a list of SkillConfig objects, or a string/Path
            pointing to a directory to scan for skills.
        backend: Optional BackendProtocol for remote filesystem discovery.
            When provided with a path, discovery is deferred to ``setup()``.
    """

    def __init__(
        self,
        *,
        skills: list[SkillConfig] | str | Path,
        backend: BackendProtocol | None = None,
        budget_percent: float | None = None,
        max_description_chars: int | None = None,
        context_window: int | None = None,
    ) -> None:
        self._backend = backend
        self._deferred_path: str | None = None

        if isinstance(skills, list):
            self._configs: list[SkillConfig] = list(skills)
        elif isinstance(skills, (str, Path)):
            if backend is not None:
                # Defer async discovery to setup()
                self._deferred_path = str(skills)
                self._configs = []
            else:
                self._configs = discover_skills_from_directory(Path(skills))
        else:
            msg = f"skills must be list[SkillConfig], str, or Path, got {type(skills).__name__}"
            raise TypeError(msg)
        self._budget_percent = budget_percent
        self._max_description_chars = max_description_chars
        self._context_window = context_window
        self._tools_cache: list[BaseTool] | None = None

    async def setup(self, **_: Any) -> None:  # type: ignore[override]
        """Run deferred async discovery if a backend path was provided."""
        if self._deferred_path is not None and self._backend is not None:
            from langchain_agentkit.extensions.skills.discovery import (
                discover_skills_from_backend,
            )

            self._configs = await discover_skills_from_backend(self._backend, self._deferred_path)
            self._deferred_path = None
            self._tools_cache = None  # Rebuild tools with discovered configs

    @property
    def configs(self) -> list[SkillConfig]:
        """The resolved skill configurations."""
        return list(self._configs)

    @property
    def state_schema(self) -> None:
        return None

    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            self._tools_cache = [
                build_skill_tool(
                    self._configs,
                    budget_percent=self._budget_percent,
                    max_description_chars=self._max_description_chars,
                    context_window=self._context_window,
                )
            ]
        return self._tools_cache

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        return _skills_system_prompt.format(
            skills_list=self._format_skills_list(),
        )

    def _format_skills_list(self) -> str:
        if not self._configs:
            return "(No skills available)"
        budget_chars = None
        if self._budget_percent is not None:
            ctx = self._context_window or 200_000
            budget_chars = int(ctx * self._budget_percent * 4)  # chars ≈ tokens * 4
        lines: list[str] = []
        total = 0
        for config in sorted(self._configs, key=lambda c: c.name):
            desc = config.description
            if self._max_description_chars is not None:
                desc = desc[: self._max_description_chars]
            line = f"- {config.name}: {desc}"
            if budget_chars is not None and total + len(line) > budget_chars:
                remaining = len(self._configs) - len(lines)
                lines.append(f"... and {remaining} more skills")
                break
            lines.append(line)
            total += len(line) + 1  # +1 for newline
        return "\n".join(lines)
