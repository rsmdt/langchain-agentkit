"""Tests for TeamState and its reducers."""

from typing import get_type_hints

from langchain_agentkit.extensions.teams.state import (
    TeamState,
    _merge_team_members,
)


class TestMergeTeamMembers:
    def test_merges_by_name_latest_wins(self):
        left = [{"name": "alice", "status": "idle"}]
        right = [{"name": "alice", "status": "working"}]

        result = _merge_team_members(left, right)

        assert len(result) == 1
        assert result[0]["status"] == "working"

    def test_new_members_appended(self):
        left = [{"name": "alice", "status": "idle"}]
        right = [{"name": "bob", "status": "idle"}]

        result = _merge_team_members(left, right)

        assert len(result) == 2
        assert result[0]["name"] == "alice"
        assert result[1]["name"] == "bob"

    def test_preserves_insertion_order(self):
        left = [
            {"name": "alice", "status": "idle"},
            {"name": "bob", "status": "idle"},
        ]
        right = [{"name": "charlie", "status": "idle"}]

        result = _merge_team_members(left, right)

        assert [m["name"] for m in result] == ["alice", "bob", "charlie"]

    def test_update_preserves_original_order(self):
        left = [
            {"name": "alice", "status": "idle"},
            {"name": "bob", "status": "idle"},
        ]
        right = [{"name": "alice", "status": "done"}]

        result = _merge_team_members(left, right)

        assert [m["name"] for m in result] == ["alice", "bob"]
        assert result[0]["status"] == "done"

    def test_empty_left(self):
        result = _merge_team_members([], [{"name": "alice", "status": "idle"}])

        assert len(result) == 1

    def test_empty_right(self):
        result = _merge_team_members([{"name": "alice", "status": "idle"}], [])

        assert len(result) == 1

    def test_both_empty(self):
        assert _merge_team_members([], []) == []

    def test_none_left(self):
        result = _merge_team_members(None, [{"name": "alice"}])

        assert len(result) == 1

    def test_none_right(self):
        result = _merge_team_members([{"name": "alice"}], None)

        assert len(result) == 1

    def test_skips_members_without_name(self):
        left = [{"status": "idle"}]
        right = [{"name": "bob", "status": "idle"}]

        result = _merge_team_members(left, right)

        assert len(result) == 1
        assert result[0]["name"] == "bob"

    def test_merge_adds_new_fields_from_right(self):
        left = [{"name": "alice", "status": "idle"}]
        right = [{"name": "alice", "agent_type": "researcher"}]

        result = _merge_team_members(left, right)

        assert result[0]["status"] == "idle"
        assert result[0]["agent_type"] == "researcher"


class TestTeamStateStructure:
    def test_has_team_members_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team_members" in hints

    def test_has_team_name_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team_name" in hints

    def test_is_total_false(self):
        assert TeamState.__total__ is False

    def test_can_instantiate_as_typed_dict(self):
        state: TeamState = {
            "team_members": [],
            "team_name": "test-team",
        }

        assert state["team_name"] == "test-team"
