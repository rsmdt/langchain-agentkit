"""LLM-driven conversation summarization for compaction.

Serializes the to-be-discarded prefix as a plaintext transcript wrapped
in ``<conversation>`` tags (so the model treats it as data, not a chat to
continue), hands it to the LLM with a structured section schema, and
returns the model's response text.

Two prompts:

* :data:`SUMMARIZATION_PROMPT` — initial summary.
* :data:`UPDATE_SUMMARIZATION_PROMPT` — merges a new transcript into an
  existing summary from a prior compaction round.

A separate :data:`TURN_PREFIX_SUMMARIZATION_PROMPT` summarizes the
initial half of a split turn so the retained suffix stays coherent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.language_models import BaseChatModel

# ---- System / user prompts --------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI coding assistant, then produce a "
    "structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""

# Tool results are truncated to this many characters when serialized for
# the summarizer — summaries rarely need full tool output verbatim.
_TOOL_RESULT_MAX_CHARS = 2000


# ---- Transcript serialization ---------------------------------------------


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype == "reasoning":
                    parts.append(str(block.get("reasoning", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _truncate(text: str, max_chars: int = _TOOL_RESULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {dropped} more characters truncated]"


def serialize_conversation(messages: Iterable[Any]) -> str:
    """Serialize messages to a plaintext transcript for summarization.

    Sending messages as-is risks the summarizer continuing the dialogue.
    Flattening to a plaintext block keeps it framed as data.
    """
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _text_from_content(msg.content)
            if text:
                out.append(f"[User]: {text}")
        elif isinstance(msg, SystemMessage):
            text = _text_from_content(msg.content)
            if text:
                out.append(f"[System]: {text}")
        elif isinstance(msg, AIMessage):
            text = _text_from_content(msg.content)
            if text:
                out.append(f"[Assistant]: {text}")
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                rendered: list[str] = []
                for call in tool_calls:
                    name = call.get("name", "tool")
                    args = call.get("args") or {}
                    try:
                        args_str = ", ".join(
                            f"{k}={json.dumps(v, default=str)}" for k, v in args.items()
                        )
                    except (TypeError, ValueError):
                        args_str = ""
                    rendered.append(f"{name}({args_str})")
                out.append("[Assistant tool calls]: " + "; ".join(rendered))
        elif isinstance(msg, ToolMessage):
            text = _text_from_content(msg.content)
            if text:
                out.append(f"[Tool result]: {_truncate(text)}")
    return "\n\n".join(out)


# ---- LLM invocation --------------------------------------------------------


async def generate_summary(
    messages: Iterable[Any],
    llm: BaseChatModel,
    *,
    reserve_tokens: int,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Ask ``llm`` to summarize the serialized transcript.

    Returns only the assistant text. Adds the previous summary (if any)
    inside ``<previous-summary>`` tags and switches to the update prompt
    so the summary accumulates cleanly across compaction rounds.
    """
    transcript = serialize_conversation(messages)
    base = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base = f"{base}\n\nAdditional focus: {custom_instructions}"

    user_text = f"<conversation>\n{transcript}\n</conversation>\n\n"
    if previous_summary:
        user_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    user_text += base

    prompt_messages = [
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    response = await llm.ainvoke(prompt_messages)
    return _text_from_content(getattr(response, "content", ""))


async def generate_turn_prefix_summary(
    messages: Iterable[Any],
    llm: BaseChatModel,
) -> str:
    """Summarize the early half of a turn that was too large to keep whole."""
    transcript = serialize_conversation(messages)
    user_text = (
        f"<conversation>\n{transcript}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
    prompt_messages = [
        SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    response = await llm.ainvoke(prompt_messages)
    return _text_from_content(getattr(response, "content", ""))
