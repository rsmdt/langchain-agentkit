"""AskUser tool — LLM-initiated structured questions via interrupt.

The tool sends a Question-based interrupt payload and returns
the user's answers as a formatted string to the LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import Field

from langchain_agentkit.extensions.hitl.types import Option, Question, StrictSchemaModel


class _QuestionInput(StrictSchemaModel):
    """A question the LLM wants to ask the user (tool-facing schema)."""

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


class _AskUserInput(StrictSchemaModel):
    """Input schema for the AskUser tool."""

    questions: list[_QuestionInput] = Field(
        description="Questions to ask the user (1-4 questions)",
        min_length=1,
        max_length=4,
    )


_ASK_USER_DESCRIPTION = """Ask the user a question and wait for their answer. Use when you must resolve ambiguity, choose between options, or get a decision before continuing. Pauses for input; it is not for statements or rhetorical questions."""


def create_ask_user_tool() -> BaseTool:
    """Create the AskUser tool for LLM-initiated human interaction.

    The tool sends structured questions via ``interrupt()`` and returns
    the user's answers as a formatted string.
    """

    def _ask_user(questions: list[dict[str, Any]]) -> str:
        """Ask the user structured questions and return their answers."""
        parsed = [
            Question(**q.model_dump()) if hasattr(q, "model_dump") else Question.model_validate(q)
            for q in questions
        ]

        response = interrupt(
            {
                "type": "question",
                "questions": [q.model_dump() for q in parsed],
            }
        )

        answers: dict[str, str | None] = {}
        if isinstance(response, dict):
            answers = response.get("answers") or {}

        parts = []
        for i, q in enumerate(parsed):
            value = answers.get(str(i))  # absent ⇒ skipped
            text = value if isinstance(value, str) and value else "(skipped)"
            parts.append(f"{q.question} → {text}")
        return "\n".join(parts)

    return StructuredTool.from_function(
        func=_ask_user,
        name="AskUser",
        description=_ASK_USER_DESCRIPTION,
        args_schema=_AskUserInput,
        handle_tool_error=True,
    )
