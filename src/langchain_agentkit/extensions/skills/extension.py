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
    discover_skills_from_backend,
    discover_skills_from_directory,
)
from langchain_agentkit.extensions.skills.tools import build_skill_tool
from langchain_agentkit.extensions.skills.types import SkillConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import BackendProtocol

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_skills_system_prompt = PromptTemplate.from_file(_PROMPT_FILE)


class SkillsExtension(Extension):
    """Extension providing the Skill tool for progressive disclosure.

    Args:
        skills: Either a list of SkillConfig objects, or a string/Path
            pointing to a directory to scan for skills.
        backend: Optional BackendProtocol for remote filesystem discovery.
    """

    def __init__(
        self,
        skills: list[SkillConfig] | str | Path,
        backend: BackendProtocol | None = None,
    ) -> None:
        if isinstance(skills, list):
            self._configs = list(skills)
        elif isinstance(skills, (str, Path)):
            if backend is not None:
                self._configs = discover_skills_from_backend(backend, str(skills))
            else:
                self._configs = discover_skills_from_directory(Path(skills))
        else:
            msg = f"skills must be list[SkillConfig], str, or Path, got {type(skills).__name__}"
            raise TypeError(msg)
        self._tools_cache: list[BaseTool] | None = None

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
            self._tools_cache = [build_skill_tool(self._configs)]
        return self._tools_cache

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        return _skills_system_prompt.format(
            skills_list=self._format_skills_list(),
        )

    def _format_skills_list(self) -> str:
        if not self._configs:
            return "(No skills available)"
        lines = []
        for config in sorted(self._configs, key=lambda c: c.name):
            lines.append(f"- **{config.name}**: {config.description}")
            lines.append(f'  -> Load via Skill("{config.name}")')
        return "\n".join(lines)
