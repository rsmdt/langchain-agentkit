"""Tests for the subagent output strategy API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.extensions.agents import (
    StrategyContext,
    SubagentOutput,
    full_history_strategy,
    last_message_strategy,
    resolve_output_strategy,
    strip_hidden_from_llm,
    trace_hidden_strategy,
)


def _ctx(prefix: str = "agentkit") -> StrategyContext:
    return StrategyContext(metadata_prefix=prefix)


def _sample_subagent_messages() -> list[AIMessage]:
    """Typical subagent trace: intermediate AIMessage(s) + final AIMessage."""
    return [
        HumanMessage(content="delegate task"),
        AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "first I think..."},
                {"type": "text", "text": "intermediate thought"},
            ],
            id="msg_1",
        ),
        AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "now I finalize..."},
                {"type": "text", "text": "final answer"},
            ],
            id="msg_2",
        ),
    ]


class TestLastMessageStrategy:
    def test_emits_single_tool_message(self):
        output = SubagentOutput(
            messages=_sample_subagent_messages(),
            structured_response=None,
            tool_call_id="call_1",
            subagent_name="researcher",
            agent_config=None,
        )
        msgs = last_message_strategy(output, _ctx())

        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == "call_1"
        assert msgs[0].name == "researcher"
        assert msgs[0].content == "final answer"

    def test_handles_no_messages(self):
        output = SubagentOutput([], None, "call_x", "researcher", None)
        msgs = last_message_strategy(output, _ctx())
        assert msgs[0].content == "(no response)"

    def test_handles_empty_final_text(self):
        output = SubagentOutput(
            messages=[AIMessage(content="")],
            structured_response=None,
            tool_call_id="call_1",
            subagent_name="researcher",
            agent_config=None,
        )
        msgs = last_message_strategy(output, _ctx())
        assert msgs[0].content == "(empty response)"


class TestFullHistoryStrategy:
    def test_emits_all_ai_messages_plus_tool_message(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_1", "researcher", None)
        msgs = full_history_strategy(output, _ctx())

        ais = [m for m in msgs if isinstance(m, AIMessage)]
        tools = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(ais) == 2
        assert len(tools) == 1
        assert tools[0].content == "final answer"
        assert tools[0].tool_call_id == "call_1"

    def test_tags_ai_messages_with_subagent_metadata(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_xyz", "researcher", None)
        msgs = full_history_strategy(output, _ctx())

        ais = [m for m in msgs if isinstance(m, AIMessage)]
        for ai in ais:
            assert ai.response_metadata["agentkit_subagent_tool_call_id"] == "call_xyz"
            assert ai.response_metadata["agentkit_subagent_name"] == "researcher"
            assert ai.response_metadata.get("agentkit_hidden_from_llm") is not True
        assert ais[-1].response_metadata["agentkit_subagent_final"] is True
        assert ais[0].response_metadata.get("agentkit_subagent_final") is not True

    def test_preserves_content_blocks(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_1", "researcher", None)
        msgs = full_history_strategy(output, _ctx())

        ais = [m for m in msgs if isinstance(m, AIMessage)]
        # Reasoning blocks survive the strategy — they're not stringified.
        reasoning_blocks = [
            b for b in ais[-1].content if isinstance(b, dict) and b.get("type") == "reasoning"
        ]
        assert reasoning_blocks
        assert reasoning_blocks[0]["reasoning"] == "now I finalize..."


class TestTraceHiddenStrategy:
    def test_tags_all_ai_messages_as_hidden(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_1", "researcher", None)
        msgs = trace_hidden_strategy(output, _ctx())

        ais = [m for m in msgs if isinstance(m, AIMessage)]
        for ai in ais:
            assert ai.response_metadata["agentkit_hidden_from_llm"] is True

    def test_tool_message_is_not_hidden(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_1", "researcher", None)
        msgs = trace_hidden_strategy(output, _ctx())

        tools = [m for m in msgs if isinstance(m, ToolMessage)]
        # ToolMessage pairs with the parent's tool_call — must remain visible.
        meta = getattr(tools[0], "response_metadata", None) or {}
        assert meta.get("agentkit_hidden_from_llm") is not True

    def test_custom_metadata_prefix(self):
        output = SubagentOutput(_sample_subagent_messages(), None, "call_1", "researcher", None)
        msgs = trace_hidden_strategy(output, _ctx(prefix="myorg"))

        ais = [m for m in msgs if isinstance(m, AIMessage)]
        for ai in ais:
            assert "myorg_hidden_from_llm" in ai.response_metadata
            assert "agentkit_hidden_from_llm" not in ai.response_metadata


class TestResolveOutputStrategy:
    def test_resolves_builtin_names(self):
        assert resolve_output_strategy("last_message") is last_message_strategy
        assert resolve_output_strategy("full_history") is full_history_strategy
        assert resolve_output_strategy("trace_hidden") is trace_hidden_strategy

    def test_passes_callable_through(self):
        def custom(_o, _c):
            return []

        assert resolve_output_strategy(custom) is custom

    def test_rejects_unknown_string(self):
        with pytest.raises(ValueError, match="Unknown output_mode"):
            resolve_output_strategy("nonsense")

    def test_rejects_non_callable_non_string(self):
        with pytest.raises(TypeError):
            resolve_output_strategy(42)


class TestStripHiddenFromLlm:
    def test_strips_tagged_messages(self):
        hidden = AIMessage(
            content="hidden",
            response_metadata={"agentkit_hidden_from_llm": True},
        )
        visible = AIMessage(content="visible")
        tool = ToolMessage(content="result", tool_call_id="call_1")

        result = strip_hidden_from_llm([visible, hidden, tool])

        assert hidden not in result
        assert visible in result
        assert tool in result

    def test_custom_prefix(self):
        hidden = AIMessage(
            content="hidden",
            response_metadata={"myorg_hidden_from_llm": True},
        )
        visible = AIMessage(content="visible")

        result = strip_hidden_from_llm([hidden, visible], metadata_prefix="myorg")
        assert hidden not in result
        assert visible in result

    def test_no_op_on_clean_list(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="ok")]
        assert strip_hidden_from_llm(msgs) == msgs

    def test_does_not_mutate_input(self):
        hidden = AIMessage(content="x", response_metadata={"agentkit_hidden_from_llm": True})
        inputs = [hidden]
        strip_hidden_from_llm(inputs)
        assert inputs == [hidden]


class TestAgentsExtensionFilter:
    """AgentsExtension.wrap_model strips hidden messages when trace_hidden."""

    def _ext(self, **kwargs):
        from langchain_agentkit.extensions.agents import AgentsExtension

        agent = MagicMock()
        agent.name = "researcher"
        agent.description = "r"
        agent.tools_inherit = False
        return AgentsExtension(agents=[agent], **kwargs)

    async def test_filters_hidden_when_trace_hidden_default(self):
        ext = self._ext()  # trace_hidden is the default
        hidden = AIMessage(content="h", response_metadata={"agentkit_hidden_from_llm": True})
        visible = AIMessage(content="v")
        captured: list = []

        async def handler(s):
            captured.append(s["messages"])
            return {}

        await ext.wrap_model(state={"messages": [visible, hidden]}, handler=handler, runtime=None)
        assert captured[0] == [visible]

    async def test_passthrough_when_last_message_strategy(self):
        """Non-hiding strategies skip the filter entirely."""
        ext = self._ext(output_mode="last_message")
        hidden = AIMessage(content="h", response_metadata={"agentkit_hidden_from_llm": True})
        captured: list = []

        async def handler(s):
            captured.append(s)
            return {}

        await ext.wrap_model(state={"messages": [hidden]}, handler=handler, runtime=None)
        # Handler received the original state dict unchanged — fast path.
        assert captured[0]["messages"] == [hidden]

    async def test_passthrough_when_full_history_strategy(self):
        ext = self._ext(output_mode="full_history")
        captured: list = []

        async def handler(s):
            captured.append(s)
            return {}

        await ext.wrap_model(
            state={"messages": [AIMessage(content="x")]}, handler=handler, runtime=None
        )
        assert len(captured) == 1

    async def test_custom_metadata_prefix(self):
        ext = self._ext(metadata_prefix="myorg")
        hidden = AIMessage(content="h", response_metadata={"myorg_hidden_from_llm": True})
        visible = AIMessage(content="v")
        captured: list = []

        async def handler(s):
            captured.append(s["messages"])
            return {}

        await ext.wrap_model(state={"messages": [hidden, visible]}, handler=handler, runtime=None)
        assert captured[0] == [visible]

    async def test_empty_messages_passthrough(self):
        ext = self._ext()
        captured: list = []

        async def handler(s):
            captured.append(s)
            return {}

        await ext.wrap_model(state={"messages": []}, handler=handler, runtime=None)
        assert captured[0] == {"messages": []}


class TestAgentsExtensionOrderingCheck:
    """setup() raises when AgentsExtension is declared BEFORE HistoryExtension
    with a message-tagging strategy. Mirrors TeamExtension's ordering check."""

    def _make_ext(self, **kwargs):
        from langchain_agentkit.extensions.agents import AgentsExtension

        agent = MagicMock()
        agent.name = "researcher"
        agent.description = "r"
        agent.tools_inherit = False
        return AgentsExtension(agents=[agent], **kwargs)

    async def test_raises_when_agents_declared_before_history(self):
        from langchain_agentkit.extensions.history import HistoryExtension

        agents_ext = self._make_ext()  # trace_hidden default
        history_ext = HistoryExtension(strategy="count", max_messages=100)
        extensions = [agents_ext, history_ext]  # wrong order

        with pytest.raises(ValueError, match="must be declared AFTER HistoryExtension"):
            await agents_ext.setup(extensions=extensions)

    async def test_passes_when_agents_declared_after_history(self):
        from langchain_agentkit.extensions.history import HistoryExtension

        agents_ext = self._make_ext()
        history_ext = HistoryExtension(strategy="count", max_messages=100)
        extensions = [history_ext, agents_ext]  # correct order

        # Should not raise.
        await agents_ext.setup(extensions=extensions)

    async def test_no_check_when_last_message_strategy(self):
        """Non-hiding strategies don't need the ordering rule — no check fires."""
        from langchain_agentkit.extensions.history import HistoryExtension

        agents_ext = self._make_ext(output_mode="last_message")
        history_ext = HistoryExtension(strategy="count", max_messages=100)
        extensions = [agents_ext, history_ext]  # would be wrong for trace_hidden

        # Should not raise — last_message doesn't tag, so ordering is irrelevant.
        await agents_ext.setup(extensions=extensions)

    async def test_no_check_when_history_absent(self):
        """Without HistoryExtension, the ordering rule is vacuous."""
        agents_ext = self._make_ext()
        # No HistoryExtension in the list.
        await agents_ext.setup(extensions=[agents_ext])
