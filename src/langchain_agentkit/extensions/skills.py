"""SkillsExtension — skill loading with two input modes.

Provides the ``Skill`` tool for progressive disclosure of skill instructions.

Two modes:

1. **Programmatic** — pass ``SkillConfig`` objects directly::

    mw = SkillsExtension(skills=[
        SkillConfig(name="research", description="...", prompt="..."),
    ])

2. **Directory discovery** — pass a directory path to scan for SKILL.md files::

    mw = SkillsExtension(skills="./skills")

    # Or with a custom backend (e.g. Daytona sandbox):
    mw = SkillsExtension(skills="/skills", backend=my_backend)

Never provides filesystem tools — that's FilesystemExtension's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import PromptTemplate

from langchain_agentkit.extension import Extension
from langchain_agentkit.frontmatter import parse_frontmatter, parse_frontmatter_string
from langchain_agentkit.tools.skill import build_skill_tool
from langchain_agentkit.types import SkillConfig
from langchain_agentkit.validate import validate_skill_config

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

    from langchain_agentkit.backend import BackendProtocol

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_skills_system_prompt = PromptTemplate.from_file(_PROMPTS_DIR / "skills_system.md")


def _strip_line_numbers(formatted: str) -> str:
    """Strip line-number prefixes from BackendProtocol.read() output.

    BackendProtocol.read() returns ``{line_num}\\t{content}`` per line.
    This strips the prefix to recover raw file content.
    """
    lines = []
    for line in formatted.splitlines(keepends=True):
        _, _, content = line.partition("\t")
        lines.append(content)
    return "".join(lines)


def _discover_skills_from_directory(
    path: Path,
) -> list[SkillConfig]:
    """Discover skills by scanning a local directory for SKILL.md files."""
    if not path.is_dir():
        return []

    configs: list[SkillConfig] = []
    seen_names: set[str] = set()

    for skill_file in sorted(path.rglob("SKILL.md")):
        try:
            result = parse_frontmatter(skill_file)
        except (OSError, UnicodeDecodeError):
            continue
        config = SkillConfig.from_frontmatter(result.metadata, result.content)

        errors = validate_skill_config(config)
        if errors:
            continue

        if config.name in seen_names:
            continue
        seen_names.add(config.name)
        configs.append(config)

    return configs


def _discover_skills_from_backend(
    backend: BackendProtocol,
    path: str,
) -> list[SkillConfig]:
    """Discover skills via a BackendProtocol by globbing for SKILL.md files."""
    matches = backend.glob("**/SKILL.md", path=path)
    configs: list[SkillConfig] = []
    seen_names: set[str] = set()

    for match in sorted(matches):
        try:
            formatted = backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            continue
        content = _strip_line_numbers(formatted)
        result = parse_frontmatter_string(content)
        config = SkillConfig.from_frontmatter(result.metadata, result.content)

        errors = validate_skill_config(config)
        if errors:
            continue

        if config.name in seen_names:
            continue
        seen_names.add(config.name)
        configs.append(config)

    return configs


class SkillsExtension(Extension):
    """Extension providing the Skill tool for progressive disclosure.

    Two input modes:

    - **List**: Pass ``SkillConfig`` objects directly.
    - **Path**: Pass a string or Path to a directory to scan for SKILL.md files.
      When ``backend`` is provided, discovery uses the backend's filesystem.
      Otherwise, scans the local OS filesystem.

    Args:
        skills: Either a list of SkillConfig objects, or a string/Path
            pointing to a directory to scan for skills.
        backend: Optional BackendProtocol for remote filesystem discovery.

    Raises:
        TypeError: If ``skills`` is not a list, str, or Path.
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
                self._configs = _discover_skills_from_backend(backend, str(skills))
            else:
                self._configs = _discover_skills_from_directory(Path(skills))
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
        """No additional state keys — skills are stateless."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """The Skill tool. Always exactly one tool."""
        if self._tools_cache is None:
            self._tools_cache = [build_skill_tool(self._configs)]
        return self._tools_cache

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str:
        """Skills system prompt with formatted skills list."""
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
