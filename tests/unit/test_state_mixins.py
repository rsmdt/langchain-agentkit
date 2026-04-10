"""Tests for TeamState and its reducers."""

from typing import get_type_hints

from langchain_agentkit.extensions.teams.state import (
    TeamMetadata,
    TeamState,
    _team_reducer,
)


class TestTeamReducer:
    def test_right_wins_when_both_present(self):
        left: TeamMetadata = {"name": "old", "members": []}
        right: TeamMetadata = {"name": "new", "members": []}

        result = _team_reducer(left, right)

        assert result == right

    def test_explicit_none_clears_existing_team(self):
        """``TeamDissolve`` returns ``{"team": None}`` — must actually clear.

        LangGraph only invokes the reducer when a node returned an
        update for this channel.  So if we see ``right=None`` here, the
        node explicitly cleared the team — it must NOT fall back to
        ``left`` or dissolve becomes a no-op.
        """
        left: TeamMetadata = {"name": "existing", "members": []}

        result = _team_reducer(left, None)

        assert result is None

    def test_replacement_wins_over_left_even_if_empty(self):
        left: TeamMetadata = {
            "name": "existing",
            "members": [{"member_name": "a", "kind": "predefined", "agent_id": "r"}],
        }
        right: TeamMetadata = {"name": "replacement", "members": []}

        result = _team_reducer(left, right)

        assert result == right
        assert result["members"] == []

    def test_initial_write_from_none(self):
        """First TeamCreate transitions ``None → metadata``."""
        right: TeamMetadata = {"name": "fresh", "members": []}

        result = _team_reducer(None, right)

        assert result == right

    def test_both_none(self):
        assert _team_reducer(None, None) is None


class TestTeamStateStructure:
    def test_has_team_key(self):
        hints = get_type_hints(TeamState, include_extras=True)

        assert "team" in hints

    def test_is_total_false(self):
        assert TeamState.__total__ is False

    def test_can_instantiate_with_team_metadata(self):
        state: TeamState = {
            "team": {
                "name": "test-team",
                "members": [
                    {
                        "member_name": "alice",
                        "kind": "predefined",
                        "agent_id": "researcher",
                    },
                ],
            },
        }

        assert state["team"]["name"] == "test-team"
        assert state["team"]["members"][0]["member_name"] == "alice"

    def test_can_instantiate_with_none_team(self):
        state: TeamState = {"team": None}

        assert state["team"] is None
