"""Subagent output strategies — shape delegation results for the parent graph.

When an ``Agent`` tool delegates to a subagent, the subagent's final state
contains a full list of messages from its ReAct loop (reasoning summaries,
intermediate AIMessages, tool interactions, final answer). This module
decides **what shape that trace takes when merged into the parent graph's
message history**.

The choice matters for three independent concerns:

1. **Parent-LLM context** — what the parent model sees on the next turn.
2. **Persistence / UI rendering** — what the checkpointer stores and the
   UI displays on reload.
3. **Tool-call pairing invariants** — every ``AIMessage(tool_calls=[...])``
   in parent state must still have a matching ``ToolMessage``.

Three built-in strategies are provided:

* ``last_message_strategy`` — emits only a ``ToolMessage`` with the
  subagent's last AIMessage text. Smallest footprint. Reasoning is
  discarded. Matches ``langgraph-supervisor``'s ``output_mode="last_message"``
  and deepagents' ``task`` tool.

* ``full_history_strategy`` — emits every message from the subagent's
  trace (tagged for origin attribution) plus the final ``ToolMessage``.
  Parent LLM sees everything. Matches supervisor's ``"full_history"``.

* ``trace_hidden_strategy`` *(default)* — emits every AIMessage from the
  subagent tagged with ``{prefix}_hidden_from_llm=True``, plus the
  terminal ``ToolMessage``. Consumers install a filter (see
  :mod:`langchain_agentkit.extensions.agents.filter`) so the parent LLM
  sees only the ``ToolMessage`` while persistence and UI retain the
  full trace. Satisfies "persist everything, inject only the last
  message".

Custom strategies are plain callables of shape
``(SubagentOutput, StrategyContext) -> list[BaseMessage]``. Pass to
``AgentsExtension(output_mode=my_strategy)``.

Conventions
-----------
Tags live on ``response_metadata`` with a configurable prefix
(``AgentsExtension(metadata_prefix="agentkit")`` by default):

- ``{prefix}_subagent_tool_call_id`` — links the message to the
  originating tool_call_id in the parent; consumers can group or scope
  by this value.
- ``{prefix}_subagent_name`` — identifies the subagent (mirrors
  ``message.name``, added on ``response_metadata`` for renderers that
  don't surface ``name``).
- ``{prefix}_subagent_final`` — ``True`` on the terminal AIMessage of a
  subagent trace, useful for "collapse to summary" retention policies.
- ``{prefix}_hidden_from_llm`` — ``True`` when the message should be
  stripped from the parent LLM's view by the filter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass(frozen=True)
class SubagentOutput:
    """Raw output produced by a subagent invocation.

    Strategies receive this and return the messages to merge into the
    parent graph's state.
    """

    messages: list[BaseMessage]
    """Complete message list from the subagent's final state."""

    structured_response: Any | None
    """Structured response if the subagent was configured with one; else None."""

    tool_call_id: str
    """The parent's tool_call_id that initiated this delegation."""

    subagent_name: str
    """Name of the subagent (from AgentConfig or 'dynamic')."""

    agent_config: Any | None
    """The AgentConfig used, if predefined; None for dynamic agents."""


@dataclass(frozen=True)
class StrategyContext:
    """Context available to a strategy beyond the raw output.

    Currently carries the metadata-tag prefix so strategies stay
    namespace-agnostic. May grow in the future (e.g. parent agent name).
    """

    metadata_prefix: str
    """Prefix for response_metadata tag keys. Defaults to 'agentkit'."""

    def tag(self, suffix: str) -> str:
        return f"{self.metadata_prefix}_{suffix}"


SubagentOutputStrategy: TypeAlias = Callable[
    ["SubagentOutput", "StrategyContext"], list[BaseMessage]
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(msg: BaseMessage) -> str:
    """Return the plain-text view of a message's content.

    Flattens a content-block list to its concatenated ``text`` fields.
    Reasoning blocks and non-text blocks are discarded from the *string*
    view — they remain intact on the AIMessages emitted by the strategy,
    which preserve ``content`` structurally.
    """
    content = msg.content
    if isinstance(content, str):
        return content.rstrip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if t:
                    parts.append(str(t))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).rstrip()
    return str(content).rstrip()


def _tag_metadata(
    msg: BaseMessage,
    ctx: StrategyContext,
    tool_call_id: str,
    subagent_name: str,
    *,
    hidden: bool,
    final: bool,
) -> dict[str, Any]:
    """Return a response_metadata dict with the subagent tags applied."""
    existing = getattr(msg, "response_metadata", None) or {}
    tagged = dict(existing)
    tagged[ctx.tag("subagent_tool_call_id")] = tool_call_id
    tagged[ctx.tag("subagent_name")] = subagent_name
    if final:
        tagged[ctx.tag("subagent_final")] = True
    if hidden:
        tagged[ctx.tag("hidden_from_llm")] = True
    return tagged


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


def last_message_strategy(
    output: SubagentOutput,
    ctx: StrategyContext,
) -> list[BaseMessage]:
    """Emit only a ToolMessage with the subagent's last-message text.

    Matches ``langgraph-supervisor``'s ``output_mode="last_message"`` and
    deepagents' ``task`` tool. Smallest footprint. Subagent reasoning
    and intermediate messages are discarded entirely.
    """
    if not output.messages:
        text = "(no response)"
    else:
        text = _extract_text(output.messages[-1]) or "(empty response)"

    return [
        ToolMessage(
            content=text,
            tool_call_id=output.tool_call_id,
            name=output.subagent_name,
        )
    ]


def full_history_strategy(
    output: SubagentOutput,
    ctx: StrategyContext,
) -> list[BaseMessage]:
    """Emit every AIMessage from the subagent plus a final ToolMessage.

    Matches ``langgraph-supervisor``'s ``output_mode="full_history"``.
    Parent LLM sees the complete subagent trace. Use when the parent
    needs to reason over the subagent's work in depth (e.g. debug,
    audit) and the context-window cost is acceptable.

    All emitted AIMessages are tagged with the subagent's origin so
    consumers can group or filter by tool_call_id if desired.
    """
    out: list[BaseMessage] = []
    ai_messages = [m for m in output.messages if isinstance(m, AIMessage)]
    for i, msg in enumerate(ai_messages):
        is_final = i == len(ai_messages) - 1
        out.append(
            msg.model_copy(
                update={
                    "response_metadata": _tag_metadata(
                        msg,
                        ctx,
                        output.tool_call_id,
                        output.subagent_name,
                        hidden=False,
                        final=is_final,
                    ),
                    "name": msg.name or output.subagent_name,
                }
            )
        )

    final_text = (
        _extract_text(ai_messages[-1]) if ai_messages else "(no response)"
    ) or "(empty response)"
    out.append(
        ToolMessage(
            content=final_text,
            tool_call_id=output.tool_call_id,
            name=output.subagent_name,
        )
    )
    return out


def trace_hidden_strategy(
    output: SubagentOutput,
    ctx: StrategyContext,
) -> list[BaseMessage]:
    """Persist every subagent AIMessage tagged hidden-from-LLM; emit plain ToolMessage.

    Default strategy. Addresses the asymmetry between persistence and
    LLM context:

    - **Persisted** (checkpoint + UI): every AIMessage from the subagent
      with its native content blocks (reasoning + text + citations)
      preserved. UIs that render ``AIMessage.content`` blocks (LangGraph
      Studio, assistant-ui, LangSmith) render reasoning as thinking on
      reload, identically to live streaming.

    - **LLM-visible**: only the final ``ToolMessage`` with the
      subagent's verbatim last-message text. The companion helper
      :func:`langchain_agentkit.extensions.agents.filter.strip_hidden_from_llm`
      removes anything tagged ``{prefix}_hidden_from_llm=True`` from the
      per-request message list; ``AgentsExtension`` wires this via its
      ``wrap_model`` hook automatically.

    ``AgentsExtension`` installs the filter for you. Advanced users who
    bypass the standard onion and call ``strip_hidden_from_llm`` directly
    get the same result — without it, the parent LLM sees the subagent
    trace as if ``full_history_strategy`` were used.
    """
    out: list[BaseMessage] = []
    ai_messages = [m for m in output.messages if isinstance(m, AIMessage)]
    for i, msg in enumerate(ai_messages):
        is_final = i == len(ai_messages) - 1
        out.append(
            msg.model_copy(
                update={
                    "response_metadata": _tag_metadata(
                        msg,
                        ctx,
                        output.tool_call_id,
                        output.subagent_name,
                        hidden=True,
                        final=is_final,
                    ),
                    "name": msg.name or output.subagent_name,
                }
            )
        )

    final_text = (
        _extract_text(ai_messages[-1]) if ai_messages else "(no response)"
    ) or "(empty response)"
    out.append(
        ToolMessage(
            content=final_text,
            tool_call_id=output.tool_call_id,
            name=output.subagent_name,
        )
    )
    return out


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


_BUILTIN_STRATEGIES = {
    "last_message": last_message_strategy,
    "full_history": full_history_strategy,
    "trace_hidden": trace_hidden_strategy,
}


def resolve_output_strategy(mode: Any) -> Any:
    """Coerce a config value to a SubagentOutputStrategy callable.

    Accepts a built-in name (``"last_message"`` / ``"full_history"`` /
    ``"trace_hidden"``) or any callable matching the strategy signature.
    """
    if callable(mode):
        return mode
    if isinstance(mode, str):
        try:
            return _BUILTIN_STRATEGIES[mode]
        except KeyError as exc:
            known = ", ".join(sorted(_BUILTIN_STRATEGIES))
            raise ValueError(
                f"Unknown output_mode '{mode}'. Expected one of: {known}, "
                f"or a callable with signature "
                f"(SubagentOutput, StrategyContext) -> list[BaseMessage]."
            ) from exc
    raise TypeError(f"output_mode must be a string or callable, got {type(mode).__name__}")
