"""Tests for ModelMetadata registry and resolver."""

from __future__ import annotations

import pytest

from langchain_agentkit.core.model_metadata import ModelMetadata
from langchain_agentkit.core.model_registry import (
    _FALLBACK_METADATA,
    DEFAULT_REGISTRY,
    register_model,
    resolve_metadata,
)


class _FakeLlm:
    """Minimal BaseChatModel stand-in carrying a model_name attribute."""

    def __init__(self, name: str, attr: str = "model_name") -> None:
        setattr(self, attr, name)


def test_exact_match_lookup() -> None:
    meta = resolve_metadata("gpt-4o")
    assert meta.name == "gpt-4o"
    assert meta.context_window == 128_000
    assert meta.provider == "openai"


def test_case_insensitive_lookup() -> None:
    meta = resolve_metadata("GPT-4O")
    assert meta.name == "gpt-4o"


def test_prefix_match_dated_id() -> None:
    # Anthropic dated model IDs should match the undated registry key.
    meta = resolve_metadata("claude-3-5-sonnet-20241022")
    assert meta.provider == "anthropic"
    assert meta.context_window == 200_000


def test_longest_prefix_wins() -> None:
    # Both gpt-4.1 and gpt-4.1-mini are registered — the longer prefix wins.
    meta = resolve_metadata("gpt-4.1-mini-2025-01-01")
    assert meta.name == "gpt-4.1-mini"


def test_extracts_model_name_from_instance() -> None:
    meta = resolve_metadata(_FakeLlm("gpt-4o"))
    assert meta.name == "gpt-4o"


def test_extracts_deployment_name_fallback() -> None:
    meta = resolve_metadata(_FakeLlm("gpt-4o", attr="deployment_name"))
    assert meta.name == "gpt-4o"


def test_fallback_when_unknown() -> None:
    meta = resolve_metadata("some-custom-model-xyz")
    assert meta is _FALLBACK_METADATA


def test_fallback_when_instance_has_no_name() -> None:
    class Bare:
        pass

    meta = resolve_metadata(Bare())
    assert meta is _FALLBACK_METADATA


def test_explicit_default_override() -> None:
    custom = ModelMetadata(name="custom", context_window=500_000)
    meta = resolve_metadata("unknown-model", default=custom)
    assert meta is custom


def test_register_model_adds_entry() -> None:
    try:
        new_meta = ModelMetadata(
            name="test-only-model-registry",
            context_window=50_000,
            provider="test",
        )
        register_model(new_meta)
        assert resolve_metadata("test-only-model-registry") is new_meta
    finally:
        DEFAULT_REGISTRY.pop("test-only-model-registry", None)


def test_frozen_dataclass() -> None:
    meta = resolve_metadata("gpt-4o")
    with pytest.raises((AttributeError, Exception)):
        meta.context_window = 1  # type: ignore[misc]


def test_kit_exposes_model_metadata() -> None:
    from langchain_agentkit.agent_kit import AgentKit

    kit = AgentKit(model=_FakeLlm("gpt-4o"))
    meta = kit.model_metadata
    assert meta.name == "gpt-4o"
    # Cached — second access returns same object.
    assert kit.model_metadata is meta


def test_kit_metadata_override() -> None:
    from langchain_agentkit.agent_kit import AgentKit

    override = ModelMetadata(name="overridden", context_window=999_999)
    kit = AgentKit(model=_FakeLlm("gpt-4o"), model_metadata=override)
    assert kit.model_metadata is override


def test_kit_metadata_string_model() -> None:
    from langchain_agentkit.agent_kit import AgentKit

    # String model with resolver — metadata comes from the string key, not
    # the resolved LLM, so kit doesn't force resolution just to get metadata.
    kit = AgentKit(model="gpt-4o", model_resolver=lambda _name: _FakeLlm("gpt-4o"))
    assert kit.model_metadata.name == "gpt-4o"
