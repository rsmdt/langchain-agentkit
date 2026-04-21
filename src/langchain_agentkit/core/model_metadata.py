"""ModelMetadata — per-model facts consumed by extensions.

Consumers (compaction, history-trimming, cost estimators) need structured
knowledge about the model they're working with: context window, output
cap, reasoning support. LangChain's chat models don't expose a uniform
surface, so this dataclass normalizes it.

Metadata is looked up via :func:`resolve_metadata` from
:mod:`langchain_agentkit.core.model_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Structured per-model facts.

    Args:
        name: Canonical model identifier (e.g. ``"gpt-4o"``).
        context_window: Maximum total tokens the model accepts (input + output).
        max_output_tokens: Cap on completion tokens, or ``None`` when unknown.
        supports_tool_calls: Whether the model supports native tool calling.
        supports_reasoning: Whether the model has a thinking / reasoning mode
            that can be toggled via provider kwargs (e.g. OpenAI o-series,
            Anthropic extended thinking, Gemini thinking).
        input_cost_per_1m: USD cost per 1M input tokens, or ``None`` when
            pricing is unknown or model is self-hosted.
        output_cost_per_1m: USD cost per 1M output tokens.
        provider: Free-form provider tag (``"openai"``, ``"anthropic"``, ...).
    """

    name: str
    context_window: int
    max_output_tokens: int | None = None
    supports_tool_calls: bool = True
    supports_reasoning: bool = False
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    provider: str = ""
