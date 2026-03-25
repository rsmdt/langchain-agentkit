"""Unit tests for eval runner — no LLM required.

Tests the trajectory matching logic, tool call extraction, and dataset format.
"""

import json

import pytest

from tests.evals.datasets import (
    ALL_DATASETS,
    _assistant_with_tools,
    _tool_call,
)
from tests.evals.eval_runner import (
    extract_tool_calls_from_openai_trajectory,
    match_tool_calls,
)

# --- Dataset Validation ---


class TestDatasetFormat:
    @pytest.mark.parametrize("name,dataset", list(ALL_DATASETS.items()))
    def test_all_entries_have_required_keys(self, name, dataset):
        for entry in dataset:
            assert "description" in entry, f"Missing 'description' in {name}"
            assert "inputs" in entry, f"Missing 'inputs' in {name}"
            assert "reference_trajectory" in entry, f"Missing 'reference_trajectory' in {name}"

    @pytest.mark.parametrize("name,dataset", list(ALL_DATASETS.items()))
    def test_inputs_are_strings(self, name, dataset):
        for entry in dataset:
            assert isinstance(entry["inputs"], str), (
                f"inputs must be str in {name}: {entry['description']}"
            )

    @pytest.mark.parametrize("name,dataset", list(ALL_DATASETS.items()))
    def test_trajectories_are_lists(self, name, dataset):
        for entry in dataset:
            assert isinstance(entry["reference_trajectory"], list), (
                f"reference_trajectory must be list in {name}"
            )

    @pytest.mark.parametrize("name,dataset", list(ALL_DATASETS.items()))
    def test_trajectory_messages_have_tool_calls(self, name, dataset):
        for entry in dataset:
            for msg in entry["reference_trajectory"]:
                assert msg["role"] == "assistant"
                assert "tool_calls" in msg


# --- Tool Call Extraction ---


class TestExtractToolCalls:
    def test_extracts_single_tool_call(self):
        trajectory = [_assistant_with_tools(_tool_call("Skill", skill_name="x"))]

        result = extract_tool_calls_from_openai_trajectory(trajectory)

        assert len(result) == 1
        assert result[0]["name"] == "Skill"
        assert result[0]["args"] == {"skill_name": "x"}

    def test_extracts_multiple_tool_calls(self):
        trajectory = [
            _assistant_with_tools(
                _tool_call("Skill", skill_name="x"),
                _tool_call("Read", file_path="/a.txt"),
            ),
        ]

        result = extract_tool_calls_from_openai_trajectory(trajectory)

        assert len(result) == 2
        assert result[0]["name"] == "Skill"
        assert result[1]["name"] == "Read"

    def test_extracts_across_multiple_messages(self):
        trajectory = [
            _assistant_with_tools(_tool_call("Skill", skill_name="x")),
            _assistant_with_tools(_tool_call("Read", file_path="/a.txt")),
        ]

        result = extract_tool_calls_from_openai_trajectory(trajectory)

        assert len(result) == 2

    def test_empty_trajectory(self):
        assert extract_tool_calls_from_openai_trajectory([]) == []

    def test_message_without_tool_calls(self):
        trajectory = [{"role": "assistant", "content": "Just text"}]

        result = extract_tool_calls_from_openai_trajectory(trajectory)

        assert result == []


# --- Trajectory Matching: Strict ---


class TestMatchToolCallsStrict:
    def test_exact_match_passes(self):
        actual = [{"name": "Skill", "args": {"skill_name": "x"}}]
        expected = [{"name": "Skill", "args": {"skill_name": "x"}}]

        passed, _ = match_tool_calls(actual, expected, mode="strict")

        assert passed

    def test_different_tool_name_fails(self):
        actual = [{"name": "Read", "args": {}}]
        expected = [{"name": "Write", "args": {}}]

        passed, comment = match_tool_calls(actual, expected, mode="strict")

        assert not passed
        assert "expected Write" in comment

    def test_different_count_fails(self):
        actual = [{"name": "Skill", "args": {}}]
        expected = [
            {"name": "Skill", "args": {}},
            {"name": "Read", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="strict")

        assert not passed

    def test_args_ignored_by_default(self):
        actual = [{"name": "Read", "args": {"file_path": "/a.txt"}}]
        expected = [{"name": "Read", "args": {"file_path": "/b.txt"}}]

        passed, _ = match_tool_calls(
            actual,
            expected,
            mode="strict",
            tool_args_mode="ignore",
        )

        assert passed

    def test_args_exact_fails_on_mismatch(self):
        actual = [{"name": "Read", "args": {"file_path": "/a.txt"}}]
        expected = [{"name": "Read", "args": {"file_path": "/b.txt"}}]

        passed, _ = match_tool_calls(
            actual,
            expected,
            mode="strict",
            tool_args_mode="exact",
        )

        assert not passed

    def test_empty_both_passes(self):
        passed, _ = match_tool_calls([], [], mode="strict")

        assert passed


# --- Trajectory Matching: Subset ---


class TestMatchToolCallsSubset:
    def test_subset_passes_when_all_expected_present(self):
        actual = [
            {"name": "Glob", "args": {}},
            {"name": "Read", "args": {}},
            {"name": "Write", "args": {}},
        ]
        expected = [{"name": "Read", "args": {}}]

        passed, _ = match_tool_calls(actual, expected, mode="subset")

        assert passed

    def test_subset_fails_when_expected_missing(self):
        actual = [{"name": "Read", "args": {}}]
        expected = [{"name": "Write", "args": {}}]

        passed, _ = match_tool_calls(actual, expected, mode="subset")

        assert not passed

    def test_subset_with_multiple_expected(self):
        actual = [
            {"name": "Skill", "args": {}},
            {"name": "Read", "args": {}},
            {"name": "Grep", "args": {}},
        ]
        expected = [
            {"name": "Skill", "args": {}},
            {"name": "Read", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="subset")

        assert passed


# --- Trajectory Matching: Unordered ---


class TestMatchToolCallsUnordered:
    def test_same_calls_different_order_passes(self):
        actual = [
            {"name": "Read", "args": {}},
            {"name": "Skill", "args": {}},
        ]
        expected = [
            {"name": "Skill", "args": {}},
            {"name": "Read", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="unordered")

        assert passed

    def test_different_count_fails(self):
        actual = [{"name": "Read", "args": {}}]
        expected = [
            {"name": "Read", "args": {}},
            {"name": "Write", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="unordered")

        assert not passed

    def test_duplicate_tools_counted_correctly(self):
        actual = [
            {"name": "Read", "args": {}},
            {"name": "Read", "args": {}},
        ]
        expected = [
            {"name": "Read", "args": {}},
            {"name": "Read", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="unordered")

        assert passed


# --- Trajectory Matching: Superset ---


class TestMatchToolCallsSuperset:
    def test_superset_passes_when_actual_subset_of_expected(self):
        actual = [{"name": "Read", "args": {}}]
        expected = [
            {"name": "Read", "args": {}},
            {"name": "Write", "args": {}},
        ]

        passed, _ = match_tool_calls(actual, expected, mode="superset")

        assert passed

    def test_superset_fails_when_actual_has_unexpected(self):
        actual = [
            {"name": "Read", "args": {}},
            {"name": "Delete", "args": {}},
        ]
        expected = [{"name": "Read", "args": {}}]

        passed, _ = match_tool_calls(actual, expected, mode="superset")

        assert not passed


# --- Edge Cases ---


class TestMatchEdgeCases:
    def test_no_expected_no_actual(self):
        passed, _ = match_tool_calls([], [])

        assert passed

    def test_no_expected_but_actual(self):
        actual = [{"name": "Read", "args": {}}]

        passed, _ = match_tool_calls(actual, [], mode="subset")

        assert not passed

    def test_expected_but_no_actual(self):
        expected = [{"name": "Read", "args": {}}]

        passed, _ = match_tool_calls([], expected, mode="subset")

        assert not passed

    def test_args_subset_matching(self):
        actual = [{"name": "Grep", "args": {"pattern": "TODO", "path": "/src"}}]
        expected = [{"name": "Grep", "args": {"pattern": "TODO"}}]

        passed, _ = match_tool_calls(
            actual,
            expected,
            mode="strict",
            tool_args_mode="subset",
        )

        assert passed

    def test_args_subset_fails_on_missing_key(self):
        actual = [{"name": "Grep", "args": {"path": "/src"}}]
        expected = [{"name": "Grep", "args": {"pattern": "TODO"}}]

        passed, _ = match_tool_calls(
            actual,
            expected,
            mode="strict",
            tool_args_mode="subset",
        )

        assert not passed


# --- Helper Tests ---


class TestHelpers:
    def test_tool_call_format(self):
        tc = _tool_call("Read", file_path="/a.txt")

        assert tc["function"]["name"] == "Read"
        assert json.loads(tc["function"]["arguments"]) == {
            "file_path": "/a.txt",
        }

    def test_assistant_with_tools_format(self):
        msg = _assistant_with_tools(
            _tool_call("Skill", skill_name="x"),
        )

        assert msg["role"] == "assistant"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "Skill"
