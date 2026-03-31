"""Shared types for the unified Question-based interrupt protocol.

Both tool approval and LLM-initiated questions use these models
to create a consistent interrupt payload for consumers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Option(BaseModel):
    """A selectable choice within a question."""

    label: str = Field(description="Display text for this choice (1-5 words)")
    description: str = Field(
        description="What this option means or what happens if chosen",
    )


class Question(BaseModel):
    """A structured question with predefined options.

    Used for both LLM-initiated questions (ask_user tool) and
    system-initiated prompts (tool call approval).
    """

    question: str = Field(description="The question to ask the user")
    header: str = Field(
        description="Short label displayed as a tag (max 12 chars)",
        max_length=12,
    )
    options: list[Option] = Field(
        description="Available choices (2-4 options)",
        min_length=2,
        max_length=4,
    )
    multi_select: bool = Field(
        default=False,
        description="Allow selecting multiple options",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for the consumer (not displayed to user)",
    )
