"""Shared types for the unified Question-based interrupt protocol.

Both tool approval and LLM-initiated questions use these models
to create a consistent interrupt payload for consumers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema


class StrictSchemaModel(BaseModel):
    """Base model whose JSON schema lists every property as ``required``.

    OpenAI strict function-calling (``bind_tools(..., strict=True)``)
    requires that the schema's ``required`` array names *every* key in
    ``properties`` — nullable types are allowed, but the key itself must be
    present. Pydantic omits fields that carry a default from ``required``,
    which makes any tool exposing such a model fail strict validation
    ("Missing '<field>' in 'required'").

    This base re-adds all properties to ``required`` at schema-generation
    time only. Instance construction still honours field defaults, and
    ``model_dump`` output (the interrupt payload) is unchanged.
    """

    @classmethod
    @override
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema = handler.resolve_ref_schema(schema)
        properties = schema.get("properties")
        if properties:
            schema["required"] = list(properties.keys())
        return schema


class Option(StrictSchemaModel):
    """A selectable choice within a question."""

    label: str = Field(description="Display text for this choice (1-5 words)")
    description: str = Field(
        description="What this option means or what happens if chosen",
    )
    preview: str | None = Field(
        default=None,
        description="Optional preview text shown when hovering or expanding the option",
    )


class Question(BaseModel):
    """A structured question with predefined options.

    Used for both LLM-initiated questions (AskUser tool) and
    system-initiated prompts (tool call approval).

    Consumers resume with ``Command(resume={"answers": {...}})`` where each
    answer is keyed by the question's position in the emitted ``questions``
    array, as a string: the answer to ``questions[0]`` is ``answers["0"]``.
    A missing index is treated as skipped.
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
