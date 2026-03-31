"""Tests for TeamState and its reducers."""

from typing import get_type_hints

from langchain_agentkit.extensions.teams.state import (
    TeamState,
    _append_messages,
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


class TestAppendMessages:
    def test_appends_right_to_left(self):
        left = [{"from": "a", "content": "hello"}]
        right = [{"from": "b", "content": "world"}]

        result = _append_messages(left, right)

        assert len(result) == 2
        assert result[0]["from"] == "a"
        assert result[1]["from"] == "b"

    def test_empty_left(self):
        result = _append_messages([], [{"content": "msg"}])

        assert len(result) == 1

    def test_empty_right(self):
        result = _append_messages([{"content": "msg"}], [])

        assert len(result) == 1

    def test_both_empty(self):
        assert _append_messages([], []) == []

    def test_none_left(self):
        result = _append_messages(None, [{"content": "msg"}])

        assert len(result) == 1

    def test_none_right(self):
        result = _append_messages([{"content": "msg"}], None)

        assert len(result) == 1

    def test_preserves_order(self):
        left = [{"id": 1}, {"id": 2}]
        right = [{"id": 3}]

        result = _append_messages(left, right)

        assert [m["id"] for m in result] == [1, 2, 3]


class TestTeamStateStructure:
    def test_has_team_members_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team_members" in hints

    def test_has_team_messages_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team_messages" in hints

    def test_has_team_name_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team_name" in hints

    def test_is_total_false(self):
        assert TeamState.__total__ is False

    def test_can_instantiate_as_typed_dict(self):
        state: TeamState = {
            "team_members": [],
            "team_messages": [],
            "team_name": "test-team",
        }

        assert state["team_name"] == "test-team"
