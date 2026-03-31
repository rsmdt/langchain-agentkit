"""Agent discovery from filesystem directories and backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extensions.agents.types import AgentConfig

if TYPE_CHECKING:
    from langchain_agentkit.backends.protocol import BackendProtocol


def _parse_comma_list(value: Any) -> list[str] | None:
    """Parse a comma-separated frontmatter value into a list of strings."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def _validate_name(name: str) -> str | None:
    """Validate a dasherized name. Returns error message or None if valid."""
    from langchain_agentkit.extensions.skills.discovery import validate_name

    return validate_name(name)


def _agent_config_from_metadata(
    metadata: dict[str, Any],
    content: str,
) -> AgentConfig | None:
    """Parse an AgentConfig from frontmatter metadata and body content."""
    name = metadata.get("name", "")
    if not name or _validate_name(name) is not None:
        return None

    max_turns_raw = metadata.get("maxTurns")
    max_turns = int(max_turns_raw) if max_turns_raw is not None else None

    return AgentConfig(
        name=name,
        description=metadata.get("description", ""),
        prompt=content,
        tools=_parse_comma_list(metadata.get("tools")),
        model=metadata.get("model") or None,
        max_turns=max_turns,
        skills=_parse_comma_list(metadata.get("skills")),
    )


def _strip_line_numbers(formatted: str) -> str:
    """Strip line-number prefixes from BackendProtocol.read() output."""
    lines = []
    for line in formatted.splitlines(keepends=True):
        _, _, content = line.partition("\t")
        lines.append(content)
    return "".join(lines)


def discover_agents_from_directory(path: Path) -> list[AgentConfig]:
    """Discover agents by scanning a local directory for .md files."""
    from langchain_agentkit.frontmatter import parse_frontmatter

    if not path.is_dir():
        return []
    agents: list[AgentConfig] = []
    seen_names: set[str] = set()
    for md_file in sorted(path.rglob("*.md")):
        try:
            result = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            continue
        agent_config = _agent_config_from_metadata(result.metadata, result.content)
        if agent_config is None or agent_config.name in seen_names:
            continue
        seen_names.add(agent_config.name)
        agents.append(agent_config)
    return agents


def discover_agents_from_backend(backend: BackendProtocol, path: str) -> list[AgentConfig]:
    """Discover agents via a BackendProtocol by globbing for .md files."""
    from langchain_agentkit.frontmatter import parse_frontmatter_string

    matches = backend.glob("**/*.md", path=path)
    agents: list[AgentConfig] = []
    seen_names: set[str] = set()
    for match in sorted(matches):
        try:
            formatted = backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            continue
        content = _strip_line_numbers(formatted)
        result = parse_frontmatter_string(content)
        agent_config = _agent_config_from_metadata(result.metadata, result.content)
        if agent_config is None or agent_config.name in seen_names:
            continue
        seen_names.add(agent_config.name)
        agents.append(agent_config)
    return agents
