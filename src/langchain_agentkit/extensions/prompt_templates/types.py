"""Data types for the prompt-template extension."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


class PromptTemplateError(ValueError):
    """Raised for invalid template metadata or bad invocation."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named, parameterized prompt fragment.

    Args:
        name: Lowercase-dasherized identifier (matches ``[a-z0-9-]{1,64}``).
        description: One-line summary shown in the system prompt and to
            the ``RunCommand`` model interface.
        body: Markdown template with ``$1`` / ``$@`` style placeholders.
        argument_hint: Optional usage hint shown to the model
            (e.g. ``"<target> [severity]"``).
        metadata: Any extra frontmatter keys, retained for downstream use.
    """

    name: str
    description: str
    body: str
    argument_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frontmatter(cls, metadata: dict[str, Any], content: str) -> PromptTemplate:
        name = str(metadata.get("name") or "").strip()
        if not _NAME_PATTERN.match(name):
            raise PromptTemplateError(
                f"Invalid template name {name!r}: must match {_NAME_PATTERN.pattern}"
            )
        description = str(metadata.get("description") or "").strip()
        if not description:
            raise PromptTemplateError(
                f"Template {name!r} is missing required 'description' frontmatter"
            )
        hint = str(
            metadata.get("argument-hint")
            or metadata.get("argumentHint")
            or metadata.get("argument_hint")
            or ""
        ).strip()
        extra = {
            k: v
            for k, v in metadata.items()
            if k not in {"name", "description", "argument-hint", "argumentHint", "argument_hint"}
        }
        return cls(
            name=name,
            description=description,
            body=content,
            argument_hint=hint,
            metadata=extra,
        )
