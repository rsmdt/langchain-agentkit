from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, override

from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.agents.llm_filter import strip_hidden_from_llm

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

FORKED_STATE_KEY = "agentkit_forked"

FORK_DELEGATION_REFUSAL = (
    "You are a delegate working on a forked conversation and cannot delegate further. "
    "Complete the task yourself."
)

_FORK_INSTRUCTION_PREAMBLE = (
    "[The messages above are the conversation you are continuing as the delegate "
    "that was just invoked. Any request in them to delegate work has already "
    "happened: you are that delegate. Delegating further is refused, so complete "
    "the task directly. Use the specific facts, figures, and identifiers already "
    "established above rather than answering generically. Your task follows.]\n\n"
)


class _ForkedState(TypedDict, total=False):
    agentkit_forked: bool


class ForkedContextExtension(Extension):
    @property
    @override
    def state_schema(self) -> type:
        return _ForkedState


def is_forked(state: dict[str, Any]) -> bool:
    return bool(state.get(FORKED_STATE_KEY))


def build_forked_state(
    parent_messages: list[BaseMessage],
    message: str,
    *,
    metadata_prefix: str,
) -> dict[str, Any]:
    history = strip_hidden_from_llm(parent_messages, metadata_prefix=metadata_prefix)
    if history and isinstance(history[-1], AIMessage) and history[-1].tool_calls:
        history.pop()
    return {
        "messages": [*history, HumanMessage(content=_FORK_INSTRUCTION_PREAMBLE + message)],
        "sender": "parent",
        FORKED_STATE_KEY: True,
    }
