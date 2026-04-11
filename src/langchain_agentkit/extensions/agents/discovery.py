"""Agent discovery from filesystem directories and backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_agentkit.extensions.agents.types import AgentConfig
from langchain_agentkit.extensions.discovery import (
    discover_from_backend,
    discover_from_directory,
)

if TYPE_CHECKING:
    from pathlib import Path

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


def _agent_config_from_metadata(
    metadata: dict[str, Any],
    content: str,
) -> AgentConfig | None:
    """Parse an AgentConfig from frontmatter metadata and body content."""
    from langchain_agentkit.extensions.skills.discovery import validate_name

    name = metadata.get("name", "")
    if not name or validate_name(name) is not None:
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


def discover_agents_from_directory(path: Path) -> list[AgentConfig]:
    """Discover agents by scanning a local directory for .md files."""
    return discover_from_directory(
        path,
        file_pattern="*.md",
        parser=_agent_config_from_metadata,
        namer=lambda c: c.name,
        label="agent",
    )


async def discover_agents_from_backend(backend: BackendProtocol, path: str) -> list[AgentConfig]:
    """Discover agents via a BackendProtocol by globbing for .md files."""
    return await discover_from_backend(
        backend,
        path,
        file_pattern="**/*.md",
        parser=_agent_config_from_metadata,
        namer=lambda c: c.name,
        label="agent",
    )
