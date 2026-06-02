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

    def test_state_schema_returns_hitl_state(self):
        from langchain_agentkit.extensions.hitl.state import HITLState

        ext = HITLExtension()

        assert ext.state_schema is HITLState

    def test_has_wrap_tool_hook(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        hooks = ext.get_all_hooks()
        assert ("wrap", "tool") in hooks

    def test_wrap_tool_discovered_as_named_hook(self):
        """wrap_tool is auto-discovered as a named hook, not legacy wrap_tool_call."""
        assert not hasattr(HITLExtension, "wrap_tool_call")
        hooks = HITLExtension._get_named_hooks()
        assert ("wrap", "tool") in hooks


# ------------------------------------------------------------------
# wrap_tool — substitutes results for non-executing decisions (no interrupt)
# ------------------------------------------------------------------


def _tool_request(tool_name="send_email", args=None, cid="call_1", decisions=None):
    """A ToolNode wrap request whose ``state`` carries the approval decisions."""
    request = MagicMock()
    request.tool_call = {"name": tool_name, "args": args or {"to": "a@b.com"}, "id": cid}
    request.state = {"hitl_decisions": decisions or {}}
    return request


class TestWrapToolSubstitution:
    """wrap_tool no longer interrupts — it applies the approval node's decisions."""

    @pytest.mark.asyncio
    async def test_no_decision_executes_normally(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request(decisions={})
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
    async def test_reject_decision_substitutes_error_without_executing(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request(
            decisions={
                "call_1": {"type": "reject", "message": "User rejected the send_email tool call."}
            }
        )
        handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "User rejected the send_email tool call."

    @pytest.mark.asyncio
    async def test_respond_decision_substitutes_success_without_executing(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request(
            decisions={"call_1": {"type": "respond", "message": "already handled"}}
        )
        handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "success"
        assert result.content == "already handled"

    @pytest.mark.asyncio
    async def test_decision_for_other_call_does_not_affect_this_one(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = _tool_request(
            cid="call_1",
            decisions={"call_2": {"type": "reject", "message": "x"}},
        )
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(state=request, handler=handler, runtime=_TEST_RUNTIME)

        handler.assert_called_once_with(request)
        assert result == expected


# ------------------------------------------------------------------
# Approval node — one batched interrupt over every gated call
# ------------------------------------------------------------------


def _state_with_calls(*calls):
    from langchain_core.messages import AIMessage

    return {"messages": [AIMessage(content="", tool_calls=list(calls))]}


def _call(name="send_email", args=None, cid="call_1"):
    return {"name": name, "args": args or {"to": "a@b.com"}, "id": cid, "type": "tool_call"}


class TestApprovalNode:
    """`_run_approval` emits a single batched interrupt and records decisions."""

    @pytest.mark.asyncio
    async def test_no_ai_message_returns_empty_decisions(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        result = await ext._run_approval({"messages": []})
        assert result == {"hitl_decisions": {}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_unconfigured_call_does_not_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": True})
        result = await ext._run_approval(_state_with_calls(_call(name="search", args={"q": "x"})))
        mock_interrupt.assert_not_called()
        assert result == {"hitl_decisions": {}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_batches_multiple_gated_calls_into_one_interrupt(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve", "1": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        await ext._run_approval(
            _state_with_calls(
                _call(cid="call_a", args={"to": "a"}),
                _call(cid="call_b", args={"to": "b"}),
            )
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

        result = await ext._run_approval(
            _state_with_calls(
                _call(cid="call_a", args={"to": "a"}),
                _call(cid="call_b", args={"to": "b"}),
            )
        )

        # call_a approved -> no recorded decision (executes); call_b rejected.
        assert "call_a" not in result["hitl_decisions"]
        assert result["hitl_decisions"]["call_b"]["type"] == "reject"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_interrupt_payload_question_format(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        await ext._run_approval(_state_with_calls(_call()))

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

        await ext._run_approval(_state_with_calls(_call()))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert [o["label"] for o in q["options"]] == ["Approve", "Edit", "Reject", "Respond"]

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_options_match_config(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": {"options": ["approve", "reject"]}})

        await ext._run_approval(_state_with_calls(_call()))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert [o["label"] for o in q["options"]] == ["Approve", "Reject"]

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_approve_records_no_decision(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await ext._run_approval(_state_with_calls(_call()))

        assert result["hitl_decisions"] == {}
        assert "messages" not in result

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_reject_records_reject_decision(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await ext._run_approval(_state_with_calls(_call()))

        decision = result["hitl_decisions"]["call_1"]
        assert decision["type"] == "reject"
        assert decision["message"] == "User rejected the send_email tool call."

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_respond_records_respond_decision(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": {"type": "respond", "message": "done"}}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await ext._run_approval(_state_with_calls(_call()))

        assert result["hitl_decisions"]["call_1"] == {"type": "respond", "message": "done"}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_edit_rewrites_call_args_on_message(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"0": {"type": "edit", "args": {"to": "new@b.com"}}}
        }
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await ext._run_approval(_state_with_calls(_call()))

        # No non-executing decision; the call's args are rewritten in place.
        assert result["hitl_decisions"] == {}
        revised = result["messages"][0].tool_calls[0]
        assert revised["args"] == {"to": "new@b.com"}
        assert revised["id"] == "call_1"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_missing_answer_defaults_to_reject(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {}}
        ext = HITLExtension(interrupt_on={"send_email": True})

        result = await ext._run_approval(_state_with_calls(_call()))

        assert result["hitl_decisions"]["call_1"]["type"] == "reject"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_single_approve_option_auto_resolves_without_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": InterruptConfig(options=["approve"])})

        result = await ext._run_approval(_state_with_calls(_call()))

        mock_interrupt.assert_not_called()
        assert result == {"hitl_decisions": {}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_single_reject_option_auto_resolves_without_interrupt(self, mock_interrupt):
        ext = HITLExtension(interrupt_on={"send_email": InterruptConfig(options=["reject"])})

        result = await ext._run_approval(_state_with_calls(_call()))

        mock_interrupt.assert_not_called()
        assert result["hitl_decisions"]["call_1"]["type"] == "reject"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_custom_question_string(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(options=["approve", "reject"], question="Send it?"),
            },
        )

        await ext._run_approval(_state_with_calls(_call()))

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

        await ext._run_approval(_state_with_calls(_call()))

        assert mock_interrupt.call_args[0][0]["questions"][0]["question"] == "Email a@b.com?"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_header_truncated_to_12_chars(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"long_tool_name_here": True})

        await ext._run_approval(_state_with_calls(_call(name="long_tool_name_here", args={})))

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert len(q["header"]) <= 12


# ------------------------------------------------------------------
# graph_modifier — pre-tools gate wiring
# ------------------------------------------------------------------


class TestGraphModifier:
    def test_registers_gate_node_when_gating(self):
        from langgraph.graph import StateGraph

        from langchain_agentkit.composition.state import AgentKitState

        ext = HITLExtension(interrupt_on={"send_email": True})
        wf = StateGraph(AgentKitState)
        wf.add_node("agent", lambda s: s)

        ext.graph_modifier(wf, "agent")

        assert "hitl_approval" in wf.nodes
        assert wf._agentkit_pretools_gate == "hitl_approval"

    def test_no_gate_node_when_not_gating(self):
        from langgraph.graph import StateGraph

        from langchain_agentkit.composition.state import AgentKitState

        ext = HITLExtension()  # AskUser only, no interrupt_on
        wf = StateGraph(AgentKitState)
        wf.add_node("agent", lambda s: s)

        ext.graph_modifier(wf, "agent")

        assert "hitl_approval" not in wf.nodes
        assert getattr(wf, "_agentkit_pretools_gate", None) is None


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
