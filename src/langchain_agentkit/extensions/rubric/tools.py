"""The ``request_review`` tool the agent calls to be graded early.

Offered to the agent only in ``user`` mode when the review policy has a
discretionary window (``max > min``). Calling it sets a flag the extension's
``after_model`` hook reads to grade at the end of the current turn instead of
waiting for the forced review at ``max``.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


def create_request_review_tool() -> object:
    """Build the ``request_review`` tool.

    Returns a tool that updates ``_rubric_review_requested`` on graph state via
    a :class:`~langgraph.types.Command`, so the next natural stop is graded.
    """

    @tool
    def request_review(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]:
        """Request a review of your work now, when you believe it satisfies the rubric.

        Use this to be graded at the end of this turn instead of waiting for the
        next scheduled review. If criteria are still unmet, you'll get specific
        feedback to address.
        """
        return Command(
            update={
                "_rubric_review_requested": True,
                "messages": [
                    ToolMessage(
                        content=(
                            "Review requested. Your work will be graded when you finish this turn."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return request_review
