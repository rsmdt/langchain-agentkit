"""DEFAULT_REGISTRY — known-model metadata table and lookup helpers.

:func:`resolve_metadata` accepts either a model name string or a
``BaseChatModel`` instance and returns the matching :class:`ModelMetadata`.
Lookup is case-insensitive with prefix fallback so dated model IDs
(e.g. ``claude-3-5-sonnet-20241022``) match their undated entry.

Register new models or override entries via :func:`register_model` — the
registry is a module-level dict and callers are expected to patch it
at import time when custom models are in play.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_agentkit.core.model_metadata import ModelMetadata

# Trailing markers we strip to reduce "dated" or "floating" IDs to a stem:
#   claude-3-5-sonnet-latest     → claude-3-5-sonnet
#   claude-3-5-sonnet-20241022   → claude-3-5-sonnet
#   gpt-4.1-mini-2025-01-01      → gpt-4.1-mini
#   claude-sonnet-4-5-preview    → claude-sonnet-4-5
_STEM_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-(latest|preview|exp|experimental)$"),
    re.compile(r"-\d{8}$"),
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),
)

_FALLBACK_METADATA = ModelMetadata(
    name="unknown",
    context_window=128_000,
    max_output_tokens=4096,
    supports_tool_calls=True,
    supports_reasoning=False,
    provider="unknown",
)


def _m(**kwargs: Any) -> ModelMetadata:
    """Compact constructor for registry entries."""
    return ModelMetadata(**kwargs)


DEFAULT_REGISTRY: dict[str, ModelMetadata] = {
    # --- OpenAI ---
    "gpt-4o": _m(
        name="gpt-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
    ),
    "gpt-4o-mini": _m(
        name="gpt-4o-mini",
        context_window=128_000,
        max_output_tokens=16_384,
        provider="openai",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    ),
    "gpt-4.1": _m(
        name="gpt-4.1",
        context_window=1_047_576,
        max_output_tokens=32_768,
        provider="openai",
        input_cost_per_1m=2.00,
        output_cost_per_1m=8.00,
    ),
    "gpt-4.1-mini": _m(
        name="gpt-4.1-mini",
        context_window=1_047_576,
        max_output_tokens=32_768,
        provider="openai",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.60,
    ),
    "gpt-4.1-nano": _m(
        name="gpt-4.1-nano",
        context_window=1_047_576,
        max_output_tokens=32_768,
        provider="openai",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
    ),
    "o1": _m(
        name="o1",
        context_window=200_000,
        max_output_tokens=100_000,
        provider="openai",
        supports_reasoning=True,
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
    ),
    "o1-mini": _m(
        name="o1-mini",
        context_window=128_000,
        max_output_tokens=65_536,
        provider="openai",
        supports_reasoning=True,
        input_cost_per_1m=3.00,
        output_cost_per_1m=12.00,
    ),
    "o3-mini": _m(
        name="o3-mini",
        context_window=200_000,
        max_output_tokens=100_000,
        provider="openai",
        supports_reasoning=True,
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
    ),
    # --- Anthropic ---
    "claude-3-5-sonnet-latest": _m(
        name="claude-3-5-sonnet-latest",
        context_window=200_000,
        max_output_tokens=8192,
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
    "claude-3-5-haiku-latest": _m(
        name="claude-3-5-haiku-latest",
        context_window=200_000,
        max_output_tokens=8192,
        provider="anthropic",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
    ),
    "claude-3-opus-latest": _m(
        name="claude-3-opus-latest",
        context_window=200_000,
        max_output_tokens=4096,
        provider="anthropic",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
    ),
    "claude-sonnet-4-5": _m(
        name="claude-sonnet-4-5",
        context_window=200_000,
        max_output_tokens=64_000,
        provider="anthropic",
        supports_reasoning=True,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
    "claude-opus-4-5": _m(
        name="claude-opus-4-5",
        context_window=200_000,
        max_output_tokens=32_000,
        provider="anthropic",
        supports_reasoning=True,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
    ),
    "claude-haiku-4-5": _m(
        name="claude-haiku-4-5",
        context_window=200_000,
        max_output_tokens=64_000,
        provider="anthropic",
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
    ),
    # --- Google ---
    "gemini-2.0-flash": _m(
        name="gemini-2.0-flash",
        context_window=1_048_576,
        max_output_tokens=8192,
        provider="google",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
    ),
    "gemini-2.5-pro": _m(
        name="gemini-2.5-pro",
        context_window=2_097_152,
        max_output_tokens=8192,
        provider="google",
        supports_reasoning=True,
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.00,
    ),
    "gemini-2.5-flash": _m(
        name="gemini-2.5-flash",
        context_window=1_048_576,
        max_output_tokens=8192,
        provider="google",
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
    ),
    # --- Meta (self-hosted) ---
    "llama-3.3-70b": _m(
        name="llama-3.3-70b",
        context_window=128_000,
        max_output_tokens=4096,
        provider="meta",
    ),
    "llama-3.1-405b": _m(
        name="llama-3.1-405b",
        context_window=128_000,
        max_output_tokens=4096,
        provider="meta",
    ),
}


def _normalize(key: str) -> str:
    return key.lower().strip()


def _stem(name: str) -> str:
    """Strip floating/date/preview suffixes to expose a stable model stem."""
    out = name
    changed = True
    while changed:
        changed = False
        for pattern in _STEM_SUFFIX_PATTERNS:
            new = pattern.sub("", out)
            if new != out:
                out = new
                changed = True
    return out


def register_model(metadata: ModelMetadata) -> None:
    """Register or replace a ``ModelMetadata`` entry in the registry.

    Keyed by the lowercase ``metadata.name``. Calling twice with the same
    name silently overwrites — the registry is flat and expects explicit
    ownership by the caller.
    """
    DEFAULT_REGISTRY[_normalize(metadata.name)] = metadata


def _lookup(name: str) -> ModelMetadata | None:
    """Exact match first, then stem match, then two-way prefix fallback."""
    norm = _normalize(name)
    if norm in DEFAULT_REGISTRY:
        return DEFAULT_REGISTRY[norm]

    # Stem match — reduce both sides to their base form (e.g.
    # ``claude-3-5-sonnet-latest`` and ``claude-3-5-sonnet-20241022`` both
    # stem to ``claude-3-5-sonnet``) and look for the longest matching key.
    norm_stem = _stem(norm)
    if norm_stem and norm_stem != norm and norm_stem in DEFAULT_REGISTRY:
        return DEFAULT_REGISTRY[norm_stem]

    best: ModelMetadata | None = None
    best_len = 0
    for key, meta in DEFAULT_REGISTRY.items():
        key_stem = _stem(key)
        if key_stem and key_stem == norm_stem and len(key_stem) > best_len:
            best = meta
            best_len = len(key_stem)
    if best is not None:
        return best

    # Directional prefix — short aliases match longer registered names or
    # vice-versa. Pick the longest match to prefer specificity.
    for key, meta in DEFAULT_REGISTRY.items():
        if norm.startswith(key) or key.startswith(norm):
            match_len = min(len(key), len(norm))
            if match_len > best_len:
                best = meta
                best_len = match_len
    return best


def _extract_model_name(llm: Any) -> str | None:
    """Extract a model name string from a ``BaseChatModel``-like instance.

    LangChain's model classes expose the name under different attributes
    depending on provider — try the common ones in order.
    """
    for attr in ("model_name", "model", "deployment_name", "deployment"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            return val
    return None


def resolve_metadata(
    key: str | Any,
    *,
    default: ModelMetadata | None = None,
) -> ModelMetadata:
    """Resolve :class:`ModelMetadata` for a model name or instance.

    Args:
        key: Either a model name string or a ``BaseChatModel``-like
            instance carrying ``model_name``/``model`` attribute.
        default: Metadata returned when no registry entry matches. When
            omitted, a conservative fallback (128k context, 4k output)
            is returned so callers never see ``None``.
    """
    name: str | None = key if isinstance(key, str) else _extract_model_name(key)
    if name:
        found = _lookup(name)
        if found is not None:
            return found
    return default if default is not None else _FALLBACK_METADATA
