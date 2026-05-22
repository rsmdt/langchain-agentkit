# ruff: noqa: N805
"""Tests for ephemeral reminder injection into the conversation tail.

The reminder channel of ``PromptComposition`` is appended to the *last*
message each step (``---``-separated, ``<reminder>``-wrapped), seen by the
LLM but never persisted to state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_agentkit import Agent
from langchain_agentkit.extension import Extension
from langchain_agentkit.graph_builder import _inject_reminder

_REMINDER = "<reminder>\n### Ext\nbody\n</reminder>"


def _make_llm() -> MagicMock:
    m = MagicMock()
    m.bind_tools = MagicMock(return_value=m)
    m.ainvoke = MagicMock(return_value=AIMessage(content="ok"))
    return m


class _ReminderExt(Extension):
    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, state, runtime=None, *, tools=frozenset()):
        return {"reminder": self._text}


class TestInjectReminderUnit:
    def test_appends_to_string_content_with_separator(self):
        state = {"messages": [HumanMessage(content="hello")]}
        out = _inject_reminder(state, _REMINDER)
        assert out["messages"][-1].content == f"hello\n\n---\n\n{_REMINDER}"

    def test_attaches_to_last_message_even_when_tool_result(self):
        state = {
            "messages": [
                HumanMessage(content="hi"),
                ToolMessage(content="grep output", tool_call_id="c1"),
            ]
        }
        out = _inject_reminder(state, _REMINDER)
        last = out["messages"][-1]
        assert last.content == f"grep output\n\n---\n\n{_REMINDER}"
        # Message type and fields preserved; earlier message untouched.
        assert isinstance(last, ToolMessage)
        assert last.tool_call_id == "c1"
        assert out["messages"][0].content == "hi"

    def test_multimodal_list_content_gets_text_block(self):
        state = {"messages": [HumanMessage(content=[{"type": "text", "text": "hi"}])]}
        out = _inject_reminder(state, _REMINDER)
        blocks = out["messages"][-1].content
        assert blocks[0] == {"type": "text", "text": "hi"}
        assert blocks[-1] == {"type": "text", "text": f"\n\n---\n\n{_REMINDER}"}

    def test_empty_reminder_is_noop(self):
        state = {"messages": [HumanMessage(content="hi")]}
        assert _inject_reminder(state, "") is state

    def test_no_messages_is_noop(self):
        state: dict = {"messages": []}
        assert _inject_reminder(state, _REMINDER) is state

    def test_does_not_mutate_original_state_or_message(self):
        msg = HumanMessage(content="hi")
        state = {"messages": [msg]}
        _inject_reminder(state, _REMINDER)
        assert msg.content == "hi"
        assert state["messages"][-1].content == "hi"


class TestReminderThroughGraph:
    async def test_llm_sees_reminder_on_tail_but_state_never_does(self):
        seen: dict = {}

        class A(Agent):
            model = _make_llm()
            extensions = [_ReminderExt("TURN-INFO")]

            async def handler(state, *, llm):
                seen["last"] = state["messages"][-1].content
                return {"messages": [AIMessage(content="done")], "sender": "a"}

        compiled = await A().compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="hello")]})

        # The LLM saw the reminder appended to the tail message.
        assert "hello\n\n---\n\n<reminder>" in seen["last"]
        assert "TURN-INFO" in seen["last"]
        # Persisted state never contains the reminder envelope.
        assert all("<reminder>" not in str(m.content) for m in result["messages"])

    async def test_reminder_rides_the_tail_into_the_tool_loop(self):
        seen: list = []
        calls = {"n": 0}

        @tool
        def noop(x: str) -> str:
            """Returns a fixed tool result."""
            return "TOOLRESULT"

        class A(Agent):
            model = _make_llm()
            tools = [noop]
            extensions = [_ReminderExt("TURN-INFO")]

            async def handler(state, *, llm):
                calls["n"] += 1
                seen.append(state["messages"][-1].content)
                if calls["n"] == 1:
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": "c1",
                                        "name": "noop",
                                        "args": {"x": "a"},
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ],
                        "sender": "a",
                    }
                return {"messages": [AIMessage(content="done")], "sender": "a"}

        compiled = await A().compile()
        await compiled.ainvoke({"messages": [HumanMessage(content="go")]})

        # Turn 1: tail is the user message.
        assert "go\n\n---\n\n<reminder>" in seen[0]
        # Turn 2 (mid-loop): tail is the tool result — reminder follows it there.
        assert "TOOLRESULT\n\n---\n\n<reminder>" in seen[1]
