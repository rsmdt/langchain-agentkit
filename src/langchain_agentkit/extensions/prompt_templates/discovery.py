"""Prompt-template discovery — scan directory or backend for ``*.md`` files."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extensions.discovery import (
    discover_from_backend,
    discover_from_directory,
)
from langchain_agentkit.extensions.prompt_templates.types import (
    PromptTemplate,
    PromptTemplateError,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_agentkit.backends.protocol import BackendProtocol

_logger = logging.getLogger(__name__)


def _parse_template(metadata: dict[str, Any], content: str) -> PromptTemplate | None:
    try:
        return PromptTemplate.from_frontmatter(metadata, content)
    except PromptTemplateError as exc:
        _logger.warning("Skipping invalid prompt template: %s", exc)
        return None


def discover_templates_from_directory(path: Path) -> list[PromptTemplate]:
    """Discover templates in a local directory (non-recursive by convention)."""
    return discover_from_directory(
        path,
        file_pattern="*.md",
        parser=_parse_template,
        namer=lambda t: t.name,
        label="prompt-template",
    )


async def discover_templates_from_backend(
    backend: BackendProtocol, path: str
) -> list[PromptTemplate]:
    """Discover templates via a :class:`BackendProtocol`."""
    return await discover_from_backend(
        backend,
        path,
        file_pattern="**/*.md",
        parser=_parse_template,
        namer=lambda t: t.name,
        label="prompt-template",
    )
