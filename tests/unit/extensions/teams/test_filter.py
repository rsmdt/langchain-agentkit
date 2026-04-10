"""Tests for ``teams/filter.py`` — the authoritative team-message filter."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_agentkit.extensions.teams.filter import (
    TEAM_KEY,
    filter_out_team_messages,
    filter_team_messages,
    is_team_tagged,
    tag_message,
    team_member_of,
)


class TestIsTeamTagged:
    def test_untagged_human_message(self):
        assert is_team_tagged(HumanMessage(content="hi")) is False

    def test_untagged_ai_message(self):
        assert is_team_tagged(AIMessage(content="hi")) is False

    def test_tagged_message(self):
        msg = HumanMessage(content="hi", additional_kwargs={TEAM_KEY: {"member": "r1"}})
        assert is_team_tagged(msg) is True

    def test_tool_message_with_tool_name_is_untagged(self):
        """ToolMessage.name holds the tool name — not a team member."""
        msg = ToolMessage(
            content="result",
            tool_call_id="t1",
            name="web_search",
        )
        assert is_team_tagged(msg) is False

    def test_tool_message_with_team_tag_is_tagged(self):
        """Authoritative tag: additional_kwargs["team"], not .name."""
        msg = ToolMessage(
            content="result",
            tool_call_id="t1",
            name="web_search",  # tool name!
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        assert is_team_tagged(msg) is True


class TestTeamMemberOf:
    def test_untagged_returns_none(self):
        assert team_member_of(HumanMessage(content="hi")) is None

    def test_tagged_returns_name(self):
        msg = AIMessage(
            content="hi",
            additional_kwargs={TEAM_KEY: {"member": "researcher"}},
        )
        assert team_member_of(msg) == "researcher"

    def test_non_string_tag_returns_none(self):
        msg = AIMessage(content="hi", additional_kwargs={TEAM_KEY: {"member": 42}})
        assert team_member_of(msg) is None


class TestFilterOutTeamMessages:
    def test_empty_list(self):
        assert filter_out_team_messages([]) == []

    def test_all_untagged(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="user"),
            AIMessage(content="assistant"),
        ]
        assert filter_out_team_messages(msgs) == msgs

    def test_removes_tagged_preserves_order(self):
        sys = SystemMessage(content="sys")
        user = HumanMessage(content="user")
        tagged = AIMessage(content="hidden", additional_kwargs={TEAM_KEY: {"member": "r1"}})
        assistant = AIMessage(content="assistant")

        result = filter_out_team_messages([sys, user, tagged, assistant])

        assert result == [sys, user, assistant]

    def test_preserves_lead_tool_call_pairs_across_intervening_team_messages(self):
        """The lead's tool-call pair stays adjacent in the filtered view.

        Physical state may interleave team-tagged messages between a
        lead's ``AIMessage(tool_calls=...)`` and its ``ToolMessage``,
        because the capture-buffer flush lands separately from the
        router's wrapped reply.  The filter must hide all tagged
        messages so the pair appears adjacent to the lead's LLM.
        """
        lead_call = AIMessage(
            content="",
            tool_calls=[{"id": "T1", "name": "TeamMessage", "args": {"to": "r1"}}],
        )
        lead_ack = ToolMessage(tool_call_id="T1", content="sent")
        team_human = HumanMessage(
            content="research",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        team_ai = AIMessage(
            content="A is ...",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )

        # Intentionally weird physical order — team messages between the pair.
        physical = [lead_call, team_human, team_ai, lead_ack]
        filtered = filter_out_team_messages(physical)

        assert filtered == [lead_call, lead_ack]


class TestFilterTeamMessages:
    def test_returns_only_named_members_slice(self):
        r1_human = HumanMessage(
            content="research",
            additional_kwargs={TEAM_KEY: {"member": "r1"}},
        )
        r1_ai = AIMessage(content="ok", additional_kwargs={TEAM_KEY: {"member": "r1"}})
        r2_human = HumanMessage(
            content="code",
            additional_kwargs={TEAM_KEY: {"member": "r2"}},
        )
        lead = HumanMessage(content="user")

        result = filter_team_messages([lead, r1_human, r2_human, r1_ai], "r1")

        assert result == [r1_human, r1_ai]

    def test_empty_when_no_match(self):
        msgs = [HumanMessage(content="user")]
        assert filter_team_messages(msgs, "r1") == []


class TestTagMessage:
    def test_tags_human_message(self):
        msg = HumanMessage(content="hi")
        tag_message(msg, "r1")

        assert msg.additional_kwargs[TEAM_KEY]["member"] == "r1"

    def test_tags_ai_message_and_sets_name(self):
        msg = AIMessage(content="hi")
        tag_message(msg, "r1")

        assert msg.additional_kwargs[TEAM_KEY]["member"] == "r1"
        assert msg.name == "r1"

    def test_preserves_existing_ai_name(self):
        msg = AIMessage(content="hi", name="custom")
        tag_message(msg, "r1")

        assert msg.additional_kwargs[TEAM_KEY]["member"] == "r1"
        assert msg.name == "custom"

    def test_idempotent(self):
        msg = HumanMessage(content="hi")
        tag_message(msg, "r1")
        tag_message(msg, "r1")

        assert msg.additional_kwargs[TEAM_KEY]["member"] == "r1"

    def test_preserves_other_additional_kwargs(self):
        msg = HumanMessage(
            content="hi",
            additional_kwargs={"existing": "value"},
        )
        tag_message(msg, "r1")

        assert msg.additional_kwargs["existing"] == "value"
        assert msg.additional_kwargs[TEAM_KEY]["member"] == "r1"

    def test_router_reply_without_team_key_passes_filter(self):
        """Router-wrapped replies have no 'team' key — lead sees them."""
        reply = HumanMessage(content="teammate's answer")

        assert not is_team_tagged(reply)
        assert filter_out_team_messages([reply]) == [reply]
