"""YAML frontmatter parser for node and skill markdown files."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FrontmatterResult:
    """Parsed frontmatter metadata and markdown body content."""

    metadata: dict[str, str] = field(default_factory=dict)
    content: str = ""


def parse_frontmatter(path: Path) -> FrontmatterResult:
    """Parse a markdown file with optional YAML frontmatter.

    Splits on ``---`` delimiters. Returns metadata dict and the body
    content with leading/trailing whitespace stripped.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    text = Path(path).read_text()

    if not text.startswith("---"):
        return FrontmatterResult(metadata={}, content=text.strip())

    parts = text.split("---", 2)
    if len(parts) < 3:
        return FrontmatterResult(metadata={}, content=text.strip())

    metadata = yaml.safe_load(parts[1]) or {}
    content = parts[2].strip()

    return FrontmatterResult(metadata=metadata, content=content)
