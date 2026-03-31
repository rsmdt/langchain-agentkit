"""ask_user tool — LLM-initiated structured questions via interrupt.

The tool sends a Question-based interrupt payload and returns
the user's answers as a formatted string to the LLM.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from langchain_agentkit.extensions.hitl.types import Option, Question


class _QuestionInput(BaseModel):
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


class _AskUserInput(BaseModel):
    """Input schema for the ask_user tool."""

    questions: list[_QuestionInput] = Field(
        description="Questions to ask the user (1-4 questions)",
        min_length=1,
        max_length=4,
    )


def create_ask_user_tool() -> BaseTool:
    """Create the ask_user tool for LLM-initiated human interaction.

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

        answers: dict[str, str] = {}
        if isinstance(response, dict):
            answers = response.get("answers", {})

        parts = []
        for q in parsed:
            answer = answers.get(q.question, "No answer provided")
            parts.append(f"{q.question} → {answer}")
        return "\n".join(parts)

    return StructuredTool.from_function(
        func=_ask_user,
        name="ask_user",
        description=(
            "Ask the user one or more structured questions with predefined "
            "options. Use this to gather preferences, clarify requirements, "
            "or get decisions on implementation choices. Each question must "
            "have 2-4 options with short labels and descriptions. The user "
            "can always provide a custom answer beyond the listed options."
        ),
        args_schema=_AskUserInput,
        handle_tool_error=True,
    )
