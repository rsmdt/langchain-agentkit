# ruff: noqa: N801, N805
"""Tests for the ReviewPolicy value object (min/max/stop + coercion)."""

from __future__ import annotations

import pytest

from langchain_agentkit.extensions.rubric.policy import HARD_CAP, ReviewPolicy


class TestCoercion:
    def test_int_expands_to_window(self):
        assert ReviewPolicy.coerce(5) == ReviewPolicy(min=1, max=5, stop=5)

    def test_one_is_every_turn(self):
        assert ReviewPolicy.coerce(1) == ReviewPolicy(min=1, max=1)

    def test_dict_maps_to_fields(self):
        assert ReviewPolicy.coerce({"min": 2, "max": 8, "stop": 4}) == ReviewPolicy(
            min=2, max=8, stop=4
        )

    def test_policy_passes_through(self):
        p = ReviewPolicy(min=2, max=3, stop=7)
        assert ReviewPolicy.coerce(p) is p

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            ReviewPolicy.coerce(True)  # type: ignore[arg-type]

    def test_other_type_rejected(self):
        with pytest.raises(TypeError, match="int, dict, or ReviewPolicy"):
            ReviewPolicy.coerce("5")  # type: ignore[arg-type]


class TestValidation:
    def test_defaults(self):
        p = ReviewPolicy()
        assert (p.min, p.max, p.stop) == (1, 1, 5)

    def test_min_above_max_rejected(self):
        with pytest.raises(ValueError, match="min <= max"):
            ReviewPolicy(min=5, max=2)

    def test_min_below_one_rejected(self):
        with pytest.raises(ValueError, match="min <= max"):
            ReviewPolicy(min=0, max=3)

    def test_max_above_cap_rejected(self):
        with pytest.raises(ValueError, match="min <= max"):
            ReviewPolicy(min=1, max=HARD_CAP + 1)

    def test_stop_below_one_rejected(self):
        with pytest.raises(ValueError, match="stop"):
            ReviewPolicy(stop=0)

    def test_stop_above_cap_rejected(self):
        with pytest.raises(ValueError, match="stop"):
            ReviewPolicy(stop=HARD_CAP + 1)

    def test_non_int_rejected(self):
        with pytest.raises(TypeError):
            ReviewPolicy(min=1.0)  # type: ignore[arg-type]

    def test_bool_field_rejected(self):
        with pytest.raises(TypeError):
            ReviewPolicy(max=True)  # type: ignore[arg-type]


class TestHasWindow:
    def test_no_window_when_min_equals_max(self):
        assert ReviewPolicy(min=2, max=2).has_window is False

    def test_window_when_max_above_min(self):
        assert ReviewPolicy(min=1, max=5).has_window is True

    def test_every_turn_has_no_window(self):
        assert ReviewPolicy(min=1, max=1).has_window is False
