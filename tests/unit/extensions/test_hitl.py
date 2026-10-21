"""Tests for HITLExtension with unified Question-based interrupt protocol."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from pydantic import ValidationError

from langchain_agentkit.extensions.hitl import HITLExtension, InterruptConfig, Option, Question

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


# ------------------------------------------------------------------
# Model validation
# ------------------------------------------------------------------


class TestQuestion:
    def test_header_max_length_enforced(self):
        with pytest.raises(ValidationError):
            Question(
                question="Test?",
                header="This is way too long for a header",
                options=[
                    Option(label="A", description="a"),
                    Option(label="B", description="b"),
                ],
            )

    def test_options_min_two(self):
        with pytest.raises(ValidationError):
            Question(
                question="Test?",
                header="Test",
                options=[Option(label="A", description="a")],
            )

    def test_options_max_four(self):
        with pytest.raises(ValidationError):
            Question(
                question="Test?",
                header="Test",
                options=[Option(label=c, description=c) for c in "ABCDE"],
            )


# ------------------------------------------------------------------
# HITLExtension init
# ------------------------------------------------------------------


class TestInit:
    def test_bool_true_expands_to_all_decisions(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        assert "send_email" in ext.interrupt_on
        assert ext.interrupt_on["send_email"].options == [
            "approve",
            "edit",
            "reject",
            "respond",
        ]

    def test_false_value_silently_ignored(self):
        """False is silently ignored — only whitelisted tools get interrupted."""
        ext = HITLExtension(interrupt_on={"search": False, "send_email": True})

        assert "search" not in ext.interrupt_on
        assert "send_email" in ext.interrupt_on

    def test_explicit_config_preserved(self):
        config = {"options": ["approve", "reject"]}
        ext = HITLExtension(interrupt_on={"send_email": config})

        assert ext.interrupt_on["send_email"].options == [
            "approve",
            "reject",
        ]

    def test_interrupt_config_object_preserved(self):
        config = InterruptConfig(
            options=["approve", "reject"],
            question="Custom description",
        )
        ext = HITLExtension(interrupt_on={"send_email": config})

        assert ext.interrupt_on["send_email"] is config

    def test_empty_interrupt_on(self):
        ext = HITLExtension(interrupt_on={})

        assert ext.interrupt_on == {}

    def test_none_interrupt_on_defaults_to_empty(self):
        ext = HITLExtension()

        assert ext.interrupt_on == {}

    def test_tools_default_is_ask_user(self):
        ext = HITLExtension()

        assert [t.name for t in ext.tools] == ["AskUser"]

    def test_tools_empty_when_explicit_empty_list(self):
        ext = HITLExtension(tools=[])

        assert ext.tools == []

    def test_tools_accepts_custom_list(self):
        from unittest.mock import MagicMock

        from langchain_core.tools import BaseTool

        custom = MagicMock(spec=BaseTool)
        ext = HITLExtension(tools=[custom])

        assert ext.tools == [custom]

    def test_both_interrupt_on_and_tools(self):
        ext = HITLExtension(
            interrupt_on={"send_email": True},
        )

        assert "send_email" in ext.interrupt_on
        assert [t.name for t in ext.tools] == ["AskUser"]


# ------------------------------------------------------------------
# Extension protocol
# ------------------------------------------------------------------


class TestExtensionProtocol:
    def test_tools_empty_when_explicit_empty_list(self):
        ext = HITLExtension(interrupt_on={"send_email": True}, tools=[])

        assert ext.tools == []

    def test_tools_contains_ask_user_by_default(self):
        ext = HITLExtension()

        assert len(ext.tools) == 1
        assert ext.tools[0].name == "AskUser"

    def test_tools_cached(self):
        ext = HITLExtension()

        first = ext.tools
        second = ext.tools

        assert first is second

    def test_prompt_returns_none(self):
        ext = HITLExtension()

        assert ext.prompt({}, _TEST_RUNTIME) is None

    def test_state_schema_is_none(self):
        assert HITLExtension().state_schema is None

    def test_has_wrap_tool_hook(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        hooks = ext.get_all_hooks()
        assert ("wrap", "tool") in hooks

    def test_wrap_tool_discovered_as_named_hook(self):
        """wrap_tool is auto-discovered as a named hook, not legacy wrap_tool_call."""
        assert not hasattr(HITLExtension, "wrap_tool_call")
        hooks = HITLExtension._get_named_hooks()
        assert ("wrap", "tool") in hooks

    def test_before_tools_discovered_as_named_hook(self):
        assert ("before", "tools") in HITLExtension._get_named_hooks()


# ------------------------------------------------------------------
# wrap_tool — returns the decision ToolMessage already in state
# ------------------------------------------------------------------


def _tool_request(tool_name="send_email", args=None, cid="call_1", messages=None):
    """A ToolNode wrap request whose ``state`` carries the conversation so far."""
    request = MagicMock()
    request.tool_call = {"name": tool_name, "args": args or {"to": "a@b.com"}, "id": cid}
    request.state = {"messages": messages or []}
    return request


class TestWrapToolSubstitution:
    @pytest.mark.asyncio
    async def test_unanswered_call_executes_normally(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request()
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_called_once_with(request)
        assert result == expected

    @pytest.mark.asyncio
    async def test_missing_state_executes_normally(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = MagicMock()
        request.tool_call = {"name": "send_email", "args": {}, "id": "call_1"}
        request.state = None
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="call_1"))

        await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_answered_call_returns_existing_message_without_executing(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        answered = ToolMessage(
            content="User rejected the send_email tool call.",
            tool_call_id="call_1",
            status="error",
            artifact={"decision": {"type": "reject"}},
        )
        request = _tool_request(messages=[answered])
        handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_not_called()
        assert result is answered

    @pytest.mark.asyncio
    async def test_answer_for_other_call_does_not_affect_this_one(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request(
            cid="call_1", messages=[ToolMessage(content="x", tool_call_id="call_2")]
        )
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_called_once_with(request)
        assert result == expected


# ------------------------------------------------------------------
# before_tools — one batched interrupt over every gated call
# ------------------------------------------------------------------


def _state_with_calls(*calls):
    from langchain_core.messages import AIMessage

    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


def _call(name="send_email", args=None, cid="call_1"):
    return {"name": name, "args": args or {"to": "a@b.com"}, "id": cid, "type": "tool_call"}


async def _approve(ext, state):
    return await ext.before_tools(state=state, runtime=_TEST_RUNTIME)


def _answers(result):
    return {
        m.tool_call_id: m for m in (result or {}).get("messages", []) if isinstance(m, ToolMessage)
    }


class TestBeforeTools:
    @pytest.mark.asyncio
    async def test_no_ai_message_returns_nothing(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        assert await _approve(ext, {"messages": []}) is None

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_unconfigured_call_does_not_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": True})
        result = await _approve(ext, _state_with_calls(_call(name="search", args={"q": "x"})))
        mock_interrupt.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_already_answered_call_is_not_asked_again(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": True})
        state = _state_with_calls(_call())
        state["messages"].append(ToolMessage(content="done", tool_call_id="call_1"))
        assert await _approve(ext, state) is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_batches_multiple_gated_calls_into_one_interrupt(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve", "1": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        await _approve(
            ext,
            _state_with_calls(
                _call(cid="call_a", args={"to": "a"}),
                _call(cid="call_b", args={"to": "b"}),
            ),
        )

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "question"
        assert len(payload["questions"]) == 2

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_index_keyed_answers_route_to_correct_call(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve", "1": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await _approve(
            ext,
            _state_with_calls(
                _call(cid="call_a", args={"to": "a"}),
                _call(cid="call_b", args={"to": "b"}),
            ),
        )

        answers = _answers(result)
        assert "call_a" not in answers
        assert answers["call_b"].status == "error"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_interrupt_payload_question_format(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        await _approve(ext, _state_with_calls(_call()))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert "question" in q
        assert "header" in q
        assert "options" in q
        assert q["context"]["tool"] == "send_email"
        assert q["context"]["args"] == {"to": "a@b.com"}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_default_options_offer_all_four_decisions(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        await _approve(ext, _state_with_calls(_call()))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert [o["label"] for o in q["options"]] == ["Approve", "Edit", "Reject", "Respond"]

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_options_match_config(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": {"options": ["approve", "reject"]}})

        await _approve(ext, _state_with_calls(_call()))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert [o["label"] for o in q["options"]] == ["Approve", "Reject"]

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_approve_answers_nothing(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        assert await _approve(ext, _state_with_calls(_call())) is None

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_reject_answers_with_error_message_and_decision_artifact(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        answer = _answers(await _approve(ext, _state_with_calls(_call())))["call_1"]

        assert answer.status == "error"
        assert answer.content == "User rejected the send_email tool call."
        assert answer.artifact == {"decision": {"type": "reject"}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_respond_answers_with_the_given_message(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": {"type": "respond", "message": "done"}}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        answer = _answers(await _approve(ext, _state_with_calls(_call())))["call_1"]

        assert answer.status == "success"
        assert answer.content == "done"
        assert answer.artifact == {"decision": {"type": "respond"}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_edit_rewrites_call_args_on_message(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"0": {"type": "edit", "args": {"to": "new@b.com"}}}
        }
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await _approve(ext, _state_with_calls(_call()))

        assert _answers(result) == {}
        revised = result["messages"][0].tool_calls[0]
        assert revised["args"] == {"to": "new@b.com"}
        assert revised["id"] == "call_1"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_missing_answer_defaults_to_reject(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        answer = _answers(await _approve(ext, _state_with_calls(_call())))["call_1"]

        assert answer.status == "error"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_single_approve_option_auto_resolves_without_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": InterruptConfig(options=["approve"])})

        result = await _approve(ext, _state_with_calls(_call()))

        mock_interrupt.assert_not_called()
        assert result is None

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_single_reject_option_auto_resolves_without_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": InterruptConfig(options=["reject"])})

        result = await _approve(ext, _state_with_calls(_call()))

        mock_interrupt.assert_not_called()
        assert _answers(result)["call_1"].status == "error"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_custom_question_string(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(options=["approve", "reject"], question="Send it?"),
            },
        )

        await _approve(ext, _state_with_calls(_call()))

        assert mock_interrupt.call_args[0][0]["questions"][0]["question"] == "Send it?"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_custom_question_callable(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "reject"],
                    question=lambda tc: f"Email {tc['args']['to']}?",
                ),
            },
        )

        await _approve(ext, _state_with_calls(_call()))

        assert mock_interrupt.call_args[0][0]["questions"][0]["question"] == "Email a@b.com?"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_header_truncated_to_12_chars(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"long_tool_name_here": True})

        await _approve(ext, _state_with_calls(_call(name="long_tool_name_here", args={})))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert len(q["header"]) <= 12

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_async_when_false_lets_the_call_run(self, mock_interrupt):
        async def never(call, state):
            return False

        ext = HITLExtension(
            interrupt_on={"send_email": {"options": ["approve", "reject"], "when": never}}
        )

        assert await _approve(ext, _state_with_calls(_call())) is None
        mock_interrupt.assert_not_called()

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_when_receives_the_call_and_state(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        seen = []

        def held(call, state):
            seen.append((call["id"], len(state["messages"])))
            return True

        ext = HITLExtension(
            interrupt_on={"send_email": {"options": ["approve", "reject"], "when": held}}
        )

        await _approve(ext, _state_with_calls(_call()))

        assert seen == [("call_1", 1)]
        mock_interrupt.assert_called_once()

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_redact_hides_args_from_the_question_only(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"0": {"type": "edit", "args": {"to": "x", "body": "secret"}}}
        }
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "edit", "reject"],
                    redact=lambda args: {k: v for k, v in args.items() if k != "body"},
                )
            }
        )

        await _approve(ext, _state_with_calls(_call(args={"to": "a", "body": "secret"})))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["context"]["args"] == {"to": "a"}
        assert "secret" not in q["question"]


class TestQuestionContext:
    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_context_callable_is_merged_into_the_question(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "reject"],
                    context=lambda call: {"kind": "connect", "toolkit": "gmail"},
                )
            }
        )

        await _approve(ext, _state_with_calls(_call()))

        context = mock_interrupt.call_args[0][0]["questions"][0]["context"]
        assert context["kind"] == "connect"
        assert context["toolkit"] == "gmail"
        assert context["tool"] == "send_email"


# ------------------------------------------------------------------
# Graph wiring — the before_tools node
# ------------------------------------------------------------------


class TestGraphWiring:
    @pytest.mark.asyncio
    async def test_gating_adds_the_before_tools_node(self):
        compiled = await _hitl_agent(HITLExtension(interrupt_on={"send_email": True}))
        assert "before_tools" in compiled.get_graph().nodes

    @pytest.mark.asyncio
    async def test_rejected_call_never_executes_and_approved_call_does(self):
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        sent: list[str] = []
        ext = HITLExtension(interrupt_on={"send_email": {"options": ["approve", "reject"]}})
        compiled = await _hitl_agent(ext, sent=sent, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t"}}

        await compiled.ainvoke({"messages": [HumanMessage(content="go")]}, config)
        result = await compiled.ainvoke(
            Command(resume={"answers": {"0": "Approve", "1": "Reject"}}), config
        )

        assert sent == ["a"]
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert sorted(m.tool_call_id for m in tool_messages) == ["call_a", "call_b"]
        rejected = next(m for m in tool_messages if m.tool_call_id == "call_b")
        assert rejected.artifact == {"decision": {"type": "reject"}}


async def _hitl_agent(ext, *, sent=None, checkpointer=None):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    from langchain_agentkit import Agent

    sink = sent if sent is not None else []

    @tool
    def send_email(to: str) -> str:
        """Send an email."""
        sink.append(to)
        return "sent"

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)

    class Mailer(Agent):
        model = llm
        tools = [send_email]
        extensions = [ext]

        async def handler(state, *, llm):  # noqa: N805
            if any(isinstance(m, ToolMessage) for m in state["messages"]):
                return {"messages": [AIMessage(content="done")], "sender": "mailer"}
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "call_a", "name": "send_email", "args": {"to": "a"}},
                            {"id": "call_b", "name": "send_email", "args": {"to": "b"}},
                        ],
                    )
                ],
                "sender": "mailer",
            }

    kwargs = {"checkpointer": checkpointer} if checkpointer is not None else {}
    return await Mailer().compile(**kwargs)


# ------------------------------------------------------------------
# AskUser tool
# ------------------------------------------------------------------


class TestAskUserTool:
    def test_tool_name_and_description(self):
        ext = HITLExtension()
        tool = ext.tools[0]

        assert tool.name == "AskUser"

    @patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt")
    def test_sends_question_interrupt_and_returns_answer(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "PostgreSQL"}}
        ext = HITLExtension()
        tool = ext.tools[0]

        result = tool.invoke(
            {
                "questions": [
                    {
                        "question": "Which database?",
                        "header": "Database",
                        "options": [
                            {"label": "PostgreSQL", "description": "Relational"},
                            {"label": "MongoDB", "description": "Document store"},
                        ],
                        "multi_select": False,
                    },
                ],
            }
        )

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "question"
        assert len(payload["questions"]) == 1
        assert "Which database?" in result
        assert "PostgreSQL" in result

    @patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt")
    def test_handles_multiple_questions(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"0": "PostgreSQL", "1": "US East"},
        }
        ext = HITLExtension()
        tool = ext.tools[0]

        result = tool.invoke(
            {
                "questions": [
                    {
                        "question": "Which database?",
                        "header": "Database",
                        "options": [
                            {"label": "PostgreSQL", "description": "Relational"},
                            {"label": "MongoDB", "description": "Document store"},
                        ],
                        "multi_select": False,
                    },
                    {
                        "question": "Which region?",
                        "header": "Region",
                        "options": [
                            {"label": "US East", "description": "Virginia"},
                            {"label": "EU West", "description": "Ireland"},
                        ],
                        "multi_select": False,
                    },
                ],
            }
        )

        assert "PostgreSQL" in result
        assert "US East" in result

    @patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt")
    def test_handles_missing_answer(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {}}
        ext = HITLExtension()
        tool = ext.tools[0]

        result = tool.invoke(
            {
                "questions": [
                    {
                        "question": "Which database?",
                        "header": "Database",
                        "options": [
                            {"label": "PostgreSQL", "description": "Relational"},
                            {"label": "MongoDB", "description": "Document store"},
                        ],
                        "multi_select": False,
                    },
                ],
            }
        )

        assert result == "Which database? → (skipped)"

    @patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt")
    def test_handles_non_dict_response(self, mock_interrupt):
        mock_interrupt.return_value = "unexpected"
        ext = HITLExtension()
        tool = ext.tools[0]

        result = tool.invoke(
            {
                "questions": [
                    {
                        "question": "Which database?",
                        "header": "Database",
                        "options": [
                            {"label": "PostgreSQL", "description": "Relational"},
                            {"label": "MongoDB", "description": "Document store"},
                        ],
                        "multi_select": False,
                    },
                ],
            }
        )

        assert result == "Which database? → (skipped)"

    @patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt")
    def test_interrupt_payload_has_no_context(self, mock_interrupt):
        """AskUser questions should not include tool-approval context."""
        mock_interrupt.return_value = {"answers": {"0": "A"}}
        ext = HITLExtension()
        tool = ext.tools[0]

        tool.invoke(
            {
                "questions": [
                    {
                        "question": "Q?",
                        "header": "Q",
                        "options": [
                            {"label": "A", "description": "a"},
                            {"label": "B", "description": "b"},
                        ],
                        "multi_select": False,
                    },
                ],
            }
        )

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["context"] is None


# ------------------------------------------------------------------
# AskUser readback — index-keyed, string-only, skip-aware
# ------------------------------------------------------------------


def _invoke_ask_user(answers):
    """Invoke the AskUser tool over two questions with the given answers."""
    from unittest.mock import patch

    ext = HITLExtension()
    tool = ext.tools[0]
    questions = [
        {
            "question": "Which region?",
            "header": "Region",
            "options": [
                {"label": "Europe", "description": "EU"},
                {"label": "US", "description": "United States"},
            ],
            "multi_select": False,
        },
        {
            "question": "Which workstreams?",
            "header": "Streams",
            "options": [
                {"label": "Market sizing", "description": "TAM"},
                {"label": "Pricing model", "description": "Price"},
            ],
            "multi_select": True,
        },
    ]
    with patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"answers": answers}
        return tool.invoke({"questions": questions})


class TestAskUserReadback:
    def test_single_question_index_keyed(self):
        from unittest.mock import patch

        ext = HITLExtension()
        tool = ext.tools[0]
        with patch("langchain_agentkit.extensions.hitl.tools.ask_user.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"answers": {"0": "Europe"}}
            result = tool.invoke(
                {
                    "questions": [
                        {
                            "question": "Which region?",
                            "header": "Region",
                            "options": [
                                {"label": "Europe", "description": "EU"},
                                {"label": "US", "description": "United States"},
                            ],
                            "multi_select": False,
                        },
                    ],
                }
            )

        assert result == "Which region? → Europe"

    def test_two_questions_multi_select_already_joined(self):
        result = _invoke_ask_user({"0": "Europe", "1": "Market sizing, Pricing model"})

        assert result == (
            "Which region? → Europe\nWhich workstreams? → Market sizing, Pricing model"
        )

    def test_missing_index_is_skipped(self):
        result = _invoke_ask_user({"0": "Europe"})

        assert result == "Which region? → Europe\nWhich workstreams? → (skipped)"

    def test_all_skipped_empty_answers(self):
        result = _invoke_ask_user({})

        assert result == ("Which region? → (skipped)\nWhich workstreams? → (skipped)")

    def test_defensive_none_value_is_skipped(self):
        result = _invoke_ask_user({"0": None})

        assert result == "Which region? → (skipped)\nWhich workstreams? → (skipped)"


# ------------------------------------------------------------------
# InterruptConfig
# ------------------------------------------------------------------


class TestInterruptConfig:
    def test_create_with_callable_question(self):
        config = InterruptConfig(
            options=["approve"],
            question=lambda tc: f"Allow {tc['name']}?",
        )

        assert callable(config.question)
        assert config.question({"name": "test"}) == "Allow test?"


# ------------------------------------------------------------------
# AskUser tool — OpenAI strict-mode schema compatibility
# ------------------------------------------------------------------


def _assert_all_object_props_required(node, path="root"):
    """Every JSON-schema object must list all of its properties as required
    (OpenAI strict function-calling rule)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            props = set(node["properties"])
            required = set(node.get("required", []))
            missing = props - required
            assert not missing, f"{path}: properties missing from 'required': {missing}"
        for key, value in node.items():
            _assert_all_object_props_required(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_all_object_props_required(value, f"{path}[{i}]")


class TestAskUserStrictSchema:
    """Regression: the AskUser tool must bind under OpenAI strict mode.

    Pydantic omits defaulted fields (Option.preview, _QuestionInput
    .multi_select) from ``required``; OpenAI strict mode 400s unless every
    property is required. StrictSchemaModel re-adds them at schema time
    without changing construction or the interrupt payload.
    """

    def test_option_preview_is_required_in_schema(self):
        from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool

        tool = create_ask_user_tool()
        schema = tool.tool_call_schema.model_json_schema()

        option_schema = schema["$defs"]["Option"]
        assert "preview" in option_schema["required"]

    def test_question_multi_select_is_required_in_schema(self):
        from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool

        tool = create_ask_user_tool()
        schema = tool.tool_call_schema.model_json_schema()

        question_schema = schema["$defs"]["_QuestionInput"]
        assert "multi_select" in question_schema["required"]

    def test_strict_openai_tool_schema_is_all_required(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool

        tool = create_ask_user_tool()
        oai = convert_to_openai_tool(tool, strict=True)

        _assert_all_object_props_required(oai["function"]["parameters"])

    def test_option_construction_still_defaults_preview_to_none(self):
        """The schema change must not force callers to pass preview."""
        opt = Option(label="A", description="a")

        assert opt.preview is None
        assert opt.model_dump() == {"label": "A", "description": "a", "preview": None}
