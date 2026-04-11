"""Generic config discovery from directories and backends.

Provides the shared glob → parse → validate → deduplicate loop used by
both skill and agent discovery.  Extension-specific modules supply a
``ConfigParser`` callback to convert frontmatter into typed configs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_agentkit.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


def discover_from_directory(
    path: Path,
    *,
    file_pattern: str,
    parser: Callable[[dict[str, Any], str], T | None],
    namer: Callable[[T], str],
    label: str,
) -> list[T]:
    """Discover configs by scanning a local directory.

    Args:
        path: Root directory to scan.
        file_pattern: Glob pattern relative to *path* (e.g. ``"SKILL.md"``, ``"*.md"``).
        parser: Converts ``(metadata, body)`` to a config or ``None``.
        namer: Extracts the dedup name from a parsed config.
        label: Human label for log messages (e.g. ``"skill"``, ``"agent"``).
    """
    from langchain_agentkit.frontmatter import parse_frontmatter

    if not path.is_dir():
        return []
    configs: list[T] = []
    seen_names: set[str] = set()
    for md_file in sorted(path.rglob(file_pattern)):
        try:
            result = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            logger.warning("Skipping unreadable %s file: %s", label, md_file)
            continue
        if not result.metadata:
            logger.warning("Skipping %s without frontmatter: %s", label, md_file)
            continue
        config = parser(result.metadata, result.content)
        if config is None:
            logger.warning("Skipping invalid %s: %s", label, md_file)
            continue
        name = namer(config)
        if name in seen_names:
            logger.warning("Skipping duplicate %s name '%s': %s", label, name, md_file)
            continue
        seen_names.add(name)
        configs.append(config)
    return configs


async def discover_from_backend(
    backend: BackendProtocol,
    path: str,
    *,
    file_pattern: str,
    parser: Callable[[dict[str, Any], str], T | None],
    namer: Callable[[T], str],
    label: str,
) -> list[T]:
    """Discover configs via a :class:`BackendProtocol`.

    Args:
        backend: Backend to read files from.
        path: Root path to scan on the backend.
        file_pattern: Glob pattern (e.g. ``"**/SKILL.md"``).
        parser: Converts ``(metadata, body)`` to a config or ``None``.
        namer: Extracts the dedup name from a parsed config.
        label: Human label for log messages.
    """
    from langchain_agentkit.frontmatter import parse_frontmatter_string

    matches = await backend.glob(file_pattern, path=path)
    configs: list[T] = []
    seen_names: set[str] = set()
    for match in sorted(matches):
        try:
            formatted = await backend.read(match, limit=100_000)
        except (FileNotFoundError, OSError):
            logger.warning("Skipping unreadable %s file: %s", label, match)
            continue
        result = parse_frontmatter_string(formatted)
        if not result.metadata:
            logger.warning("Skipping %s without frontmatter: %s", label, match)
            continue
        config = parser(result.metadata, result.content)
        if config is None:
            logger.warning("Skipping invalid %s: %s", label, match)
            continue
        name = namer(config)
        if name in seen_names:
            logger.warning("Skipping duplicate %s name '%s': %s", label, name, match)
            continue
        seen_names.add(name)
        configs.append(config)
    return configs
