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
from typing import TYPE_CHECKING, Any, override

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.skills.discovery import (
    discover_skills_from_directory,
)
from langchain_agentkit.extensions.skills.tools import (
    build_skill_tool,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backends.protocol import FilesystemProtocol
    from langchain_agentkit.extensions.skills.types import SkillConfig

_PROMPT_FILE = Path(__file__).parent / "prompt.md"
_skills_system_prompt = PromptTemplate.from_file(_PROMPT_FILE)


class SkillsExtension(Extension):
    """Extension providing the Skill tool for progressive disclosure.

    Args:
        skills: Either a list of SkillConfig objects, or a string/Path
            pointing to a directory to scan for skills.
        backend: Optional FilesystemProtocol for remote filesystem discovery.
            When provided with a path, discovery is deferred to ``setup()``.
        max_description_chars: Per-skill description cap in the system-prompt
            roster (default 250). Set to ``0`` to disable the cap.
    """

    def __init__(
        self,
        *,
        skills: list[SkillConfig] | str | Path,
        backend: FilesystemProtocol | None = None,
        max_description_chars: int = 250,
        tools: Sequence[BaseTool] | None = None,
    ) -> None:
        self._backend = backend
        self._deferred_path: str | None = None

        if isinstance(skills, list):
            self._configs: list[SkillConfig] = list(skills)
        elif isinstance(skills, (str, Path)):
            if backend is not None:
                if tools is not None:
                    raise ValueError(
                        "tools= cannot be combined with backend-based skill discovery. "
                        "Either pass explicit SkillConfigs via skills=[...] and omit "
                        "backend, or omit tools= so discovery can build the default "
                        "Skill tool from discovered configs."
                    )
                # Defer async discovery to setup()
                self._deferred_path = str(skills)
                self._configs = []
            else:
                self._configs = discover_skills_from_directory(Path(skills))
        else:
            msg = f"skills must be list[SkillConfig], str, or Path, got {type(skills).__name__}"
            raise TypeError(msg)
        self._max_description_chars = max_description_chars
        self._custom_tools: tuple[BaseTool, ...] | None = (
            tuple(tools) if tools is not None else None
        )
        self._tools_cache: list[BaseTool] | None = None

    @override
    async def setup(self, **_: Any) -> None:  # type: ignore[override]
        if self._deferred_path is not None and self._backend is not None:
            from langchain_agentkit.extensions.skills.discovery import (
                discover_skills_from_backend,
            )

            self._configs = await discover_skills_from_backend(self._backend, self._deferred_path)
            self._deferred_path = None
            self._tools_cache = None  # Rebuild tools with discovered configs

    @property
    def configs(self) -> list[SkillConfig]:
        return list(self._configs)

    @property
    @override
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            if self._custom_tools is not None:
                self._tools_cache = list(self._custom_tools)
            elif self._configs:
                self._tools_cache = [build_skill_tool(self._configs)]
            else:
                # No skills → no Skill tool. The extension is inert when empty.
                self._tools_cache = []
        return self._tools_cache

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str | None:
        # No skills → contribute nothing, as if the extension were not added.
        if not self._configs:
            return None
        # The skill roster is fixed at construction, so it is static content:
        # it belongs in the cacheable system prompt, not the per-turn reminder.
        return _skills_system_prompt.format(skills_list=self._format_skills_list())

    def _format_skills_list(self) -> str:
        lines: list[str] = []
        for config in sorted(self._configs, key=lambda c: c.name):
            desc = config.description
            # Cap each entry so verbose descriptions don't inflate the cached
            # system prompt; ``max_description_chars=0`` disables the cap.
            cap = self._max_description_chars
            if cap and len(desc) > cap:
                desc = desc[: cap - 1] + "…"
            lines.append(f"- {config.name}: {desc}")
        return "\n".join(lines)
