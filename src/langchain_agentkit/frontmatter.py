"""YAML frontmatter parser for node and skill markdown files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrontmatterResult:
    """Parsed frontmatter metadata and markdown body content."""

    metadata: dict[str, str] = field(default_factory=dict)
    content: str = ""


def parse_frontmatter_string(text: str) -> FrontmatterResult:
    """Parse a string with optional YAML frontmatter.

    Splits on ``---`` delimiters. Returns metadata dict and the body
    content with leading/trailing whitespace stripped.  Malformed YAML
    is logged as a warning and treated as missing frontmatter.
    """
    if not text.startswith("---"):
        return FrontmatterResult(metadata={}, content=text.strip())

    parts = text.split("---", 2)
    if len(parts) < 3:
        return FrontmatterResult(metadata={}, content=text.strip())

    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        logger.warning("Malformed YAML frontmatter, treating as plain content: %s", exc)
        return FrontmatterResult(metadata={}, content=parts[2].strip())

    if not isinstance(metadata, dict):
        kind = type(metadata).__name__
        logger.warning("Frontmatter YAML is not a mapping (got %s), ignoring", kind)
        return FrontmatterResult(metadata={}, content=parts[2].strip())

    content = parts[2].strip()

    return FrontmatterResult(metadata=metadata, content=content)


def parse_frontmatter(path: Path) -> FrontmatterResult:
    """Parse a markdown file with optional YAML frontmatter.

    Splits on ``---`` delimiters. Returns metadata dict and the body
    content with leading/trailing whitespace stripped.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    text = Path(path).read_text()
    return parse_frontmatter_string(text)
