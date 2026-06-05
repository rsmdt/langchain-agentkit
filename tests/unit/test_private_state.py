# ruff: noqa: N801, N805
"""Tests for the ``PrivateStateAttr`` marker and public-schema derivation."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_agentkit.composition.state import (
    AgentKitState,
    PrivateStateAttr,
    _has_private_marker,
    public_state_schema,
)


class TestPrivateMarker:
    def test_marker_is_not_callable(self):
        # LangGraph treats any callable in Annotated metadata as a reducer, so
        # the marker MUST be a non-callable sentinel.
        assert not callable(PrivateStateAttr)

    def test_detects_direct_annotated(self):
        assert _has_private_marker(Annotated[int, PrivateStateAttr]) is True

    def test_detects_through_notrequired_wrapper(self):
        assert _has_private_marker(NotRequired[Annotated[str, PrivateStateAttr]]) is True

    def test_detects_alongside_a_reducer(self):
        def _reducer(a: Any, b: Any) -> Any:
            return b

        assert _has_private_marker(Annotated[int, PrivateStateAttr, _reducer]) is True

    def test_plain_type_is_not_private(self):
        assert _has_private_marker(int) is False

    def test_annotated_without_marker_is_not_private(self):
        assert _has_private_marker(Annotated[int, "some other meta"]) is False


def _merge(a: list, b: list) -> list:
    """Module-level reducer so string annotations resolve via get_type_hints."""
    return a + b


class _SampleState(TypedDict, total=False):
    public_key: str
    messages: Annotated[list, _merge]
    _secret: Annotated[int, PrivateStateAttr]


class TestPublicStateSchema:
    def test_returns_none_when_nothing_private(self):
        # AgentKitState has no PrivateStateAttr keys.
        assert public_state_schema(AgentKitState) is None

    def test_excludes_private_keeps_public(self):
        public = public_state_schema(_SampleState)
        assert public is not None
        keys = set(public.__annotations__)
        assert "public_key" in keys
        assert "messages" in keys
        assert "_secret" not in keys

    def test_preserves_reducer_on_kept_key(self):
        public = public_state_schema(_SampleState)
        assert public is not None
        # The kept key still carries its reducer in the annotation metadata.
        assert _merge in public.__annotations__["messages"].__metadata__


class TestComposedRubricState:
    def test_rubric_bookkeeping_excluded_public_rubric_kept(self):
        from langchain_agentkit import AgentKit
        from langchain_agentkit.extensions.rubric import RubricExtension

        kit = AgentKit(extensions=[RubricExtension(model="stub:test")])
        public = public_state_schema(kit.state_schema)
        assert public is not None
        keys = set(public.__annotations__)
        # Caller-facing keys remain.
        assert "rubric" in keys
        assert "messages" in keys
        # Internal bookkeeping is hidden.
        for private in (
            "_rubric_status",
            "_rubric_iterations",
            "_rubric_evaluations",
            "_current_grading_run_id",
            "_active_rubric",
        ):
            assert private not in keys
        # ...but the full composed schema still declares them as channels.
        assert "_rubric_status" in kit.state_schema.__annotations__
