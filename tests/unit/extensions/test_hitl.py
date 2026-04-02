"""Tests for HITLExtension with unified Question-based interrupt protocol."""

from unittest.mock import MagicMock, patch

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


class TestOption:
    def test_create(self):
        opt = Option(label="PostgreSQL", description="Relational DB")

        assert opt.label == "PostgreSQL"
        assert opt.description == "Relational DB"

    def test_requires_label_and_description(self):
        with pytest.raises(ValidationError):
            Option(label="Only label")  # type: ignore[call-arg]


class TestQuestion:
    def test_create_minimal(self):
        q = Question(
            question="Which database?",
            header="Database",
            options=[
                Option(label="PostgreSQL", description="Relational"),
                Option(label="MongoDB", description="Document store"),
            ],
        )

        assert q.question == "Which database?"
        assert q.header == "Database"
        assert len(q.options) == 2
        assert q.multi_select is False
        assert q.context is None

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

    def test_context_preserved(self):
        q = Question(
            question="Test?",
            header="Test",
            options=[
                Option(label="A", description="a"),
                Option(label="B", description="b"),
            ],
            context={"tool": "send_email", "args": {"to": "a@b.com"}},
        )

        assert q.context == {"tool": "send_email", "args": {"to": "a@b.com"}}

    def test_multi_select(self):
        q = Question(
            question="Which features?",
            header="Features",
            options=[
                Option(label="Auth", description="Authentication"),
                Option(label="Logs", description="Logging"),
            ],
            multi_select=True,
        )

        assert q.multi_select is True

    def test_model_dump_roundtrip(self):
        q = Question(
            question="Test?",
            header="Test",
            options=[
                Option(label="A", description="a"),
                Option(label="B", description="b"),
            ],
            context={"key": "value"},
        )

        restored = Question.model_validate(q.model_dump())

        assert restored == q


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

    def test_tools_default_false(self):
        ext = HITLExtension()

        assert ext._provide_tools is False

    def test_tools_enabled(self):
        ext = HITLExtension(tools=True)

        assert ext._provide_tools is True

    def test_both_interrupt_on_and_tools(self):
        ext = HITLExtension(
            interrupt_on={"send_email": True},
            tools=True,
        )

        assert "send_email" in ext.interrupt_on
        assert ext._provide_tools is True


# ------------------------------------------------------------------
# Extension protocol
# ------------------------------------------------------------------


class TestExtensionProtocol:
    def test_tools_empty_when_ask_user_disabled(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        assert ext.tools == []

    def test_tools_contains_ask_user_when_enabled(self):
        ext = HITLExtension(tools=True)

        assert len(ext.tools) == 1
        assert ext.tools[0].name == "ask_user"

    def test_tools_cached(self):
        ext = HITLExtension(tools=True)

        first = ext.tools
        second = ext.tools

        assert first is second

    def test_prompt_returns_none(self):
        ext = HITLExtension(tools=True)

        assert ext.prompt({}, _TEST_RUNTIME) is None

    def test_state_schema_returns_none(self):
        ext = HITLExtension()

        assert ext.state_schema is None

    def test_has_wrap_tool_call_method(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        assert callable(getattr(ext, "wrap_tool_call", None))


# ------------------------------------------------------------------
# wrap_tool_call — auto-approved tools
# ------------------------------------------------------------------


class TestWrapToolCallAutoApproved:
    def test_unconfigured_tool_executes_normally(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        mock_request = MagicMock()
        mock_request.tool_call = {
            "name": "search",
            "args": {"q": "test"},
            "id": "call_1",
        }
        expected = ToolMessage(content="result", tool_call_id="call_1")
        mock_execute = MagicMock(return_value=expected)

        result = ext.wrap_tool_call(mock_request, mock_execute)

        mock_execute.assert_called_once_with(mock_request)
        assert result == expected

    def test_single_approve_decision_auto_executes(self):
        """Only one allowed decision (approve) — no point in asking."""
        ext = HITLExtension(
            interrupt_on={
                "search": InterruptConfig(options=["approve"]),
            },
        )
        mock_request = MagicMock()
        mock_request.tool_call = {
            "name": "search",
            "args": {"q": "test"},
            "id": "call_1",
        }
        expected = ToolMessage(content="result", tool_call_id="call_1")
        mock_execute = MagicMock(return_value=expected)

        result = ext.wrap_tool_call(mock_request, mock_execute)

        mock_execute.assert_called_once_with(mock_request)
        assert result == expected


# ------------------------------------------------------------------
# wrap_tool_call — interrupt flow
# ------------------------------------------------------------------


class TestWrapToolCallInterrupt:
    def _make_request(self, tool_name="send_email", args=None):
        mock = MagicMock()
        mock.tool_call = {
            "name": tool_name,
            "args": args or {"to": "a@b.com"},
            "id": "call_1",
        }
        return mock

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_interrupt_payload_uses_question_format(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"Tool: send_email\nArgs: {'to': 'a@b.com'}": "Approve"},
        }
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        ext.wrap_tool_call(request, MagicMock())

        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "question"
        assert len(payload["questions"]) == 1
        q = payload["questions"][0]
        assert "question" in q
        assert "header" in q
        assert "options" in q
        assert q["context"]["tool"] == "send_email"
        assert q["context"]["args"] == {"to": "a@b.com"}

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_options_match_options(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"Tool: send_email\nArgs: {'to': 'a@b.com'}": "Approve"},
        }
        ext = HITLExtension(
            interrupt_on={
                "send_email": {"options": ["approve", "reject"]},
            },
        )
        request = self._make_request()

        ext.wrap_tool_call(request, MagicMock())

        q = mock_interrupt.call_args[0][0]["questions"][0]
        labels = [o["label"] for o in q["options"]]
        assert labels == ["Approve", "Reject"]

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_approve_executes_tool(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {"answers": {description: "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        mock_execute = MagicMock(return_value=expected)

        result = ext.wrap_tool_call(request, mock_execute)

        mock_execute.assert_called_once_with(request)
        assert result == expected

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_reject_returns_error(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {
            "answers": {description: "Reject"},
            "message": "Not appropriate",
        }
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_execute = MagicMock()

        result = ext.wrap_tool_call(request, mock_execute)

        mock_execute.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Not appropriate" in result.content

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_reject_default_message(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {"answers": {description: "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        result = ext.wrap_tool_call(request, MagicMock())

        assert "User rejected send_email" in result.content

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_edit_modifies_args_and_executes(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {
            "answers": {description: "Edit"},
            "edited_args": {"to": "new@b.com"},
        }
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        mock_execute = MagicMock(return_value=expected)

        result = ext.wrap_tool_call(request, mock_execute)

        mock_execute.assert_called_once()
        request.override.assert_called_once()
        override_kwargs = request.override.call_args[1]
        assert override_kwargs["tool_call"]["args"] == {"to": "new@b.com"}
        assert result == expected

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_edit_without_edited_args_uses_original(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {"answers": {description: "Edit"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_execute = MagicMock()

        ext.wrap_tool_call(request, mock_execute)

        override_kwargs = request.override.call_args[1]
        assert override_kwargs["tool_call"]["args"] == {"to": "a@b.com"}

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_invalid_answer_returns_error(self, mock_interrupt):
        description = "Tool: send_email\nArgs: {'to': 'a@b.com'}"
        mock_interrupt.return_value = {"answers": {description: "InvalidChoice"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_execute = MagicMock()

        result = ext.wrap_tool_call(request, mock_execute)

        mock_execute.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Invalid answer" in result.content

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_empty_response_returns_error(self, mock_interrupt):
        mock_interrupt.return_value = {}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        result = ext.wrap_tool_call(request, MagicMock())

        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_non_dict_response_returns_error(self, mock_interrupt):
        mock_interrupt.return_value = "unexpected string"
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        result = ext.wrap_tool_call(request, MagicMock())

        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_custom_description_string(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"Send email to user?": "Approve"},
        }
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "reject"],
                    question="Send email to user?",
                ),
            },
        )
        request = self._make_request()

        ext.wrap_tool_call(request, MagicMock())

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["question"] == "Send email to user?"

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_custom_question_callable(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"Email a@b.com?": "Approve"},
        }
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "reject"],
                    question=lambda tc: f"Email {tc['args']['to']}?",
                ),
            },
        )
        request = self._make_request()

        ext.wrap_tool_call(request, MagicMock())

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["question"] == "Email a@b.com?"

    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    def test_header_truncated_to_12_chars(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {}}
        ext = HITLExtension(
            interrupt_on={"long_tool_name_here": True},
        )
        request = MagicMock()
        request.tool_call = {
            "name": "long_tool_name_here",
            "args": {},
            "id": "call_1",
        }

        ext.wrap_tool_call(request, MagicMock())

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert len(q["header"]) <= 12


# ------------------------------------------------------------------
# ask_user tool
# ------------------------------------------------------------------


class TestAskUserTool:
    def test_tool_name_and_description(self):
        ext = HITLExtension(tools=True)
        tool = ext.tools[0]

        assert tool.name == "ask_user"
        assert "ask the user a question" in tool.description.lower()

    @patch("langchain_agentkit.extensions.hitl.tools.interrupt")
    def test_sends_question_interrupt_and_returns_answer(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {"Which database?": "PostgreSQL"},
        }
        ext = HITLExtension(tools=True)
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

    @patch("langchain_agentkit.extensions.hitl.tools.interrupt")
    def test_handles_multiple_questions(self, mock_interrupt):
        mock_interrupt.return_value = {
            "answers": {
                "Which database?": "PostgreSQL",
                "Which region?": "US East",
            },
        }
        ext = HITLExtension(tools=True)
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

    @patch("langchain_agentkit.extensions.hitl.tools.interrupt")
    def test_handles_missing_answer(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {}}
        ext = HITLExtension(tools=True)
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

        assert "No answer provided" in result

    @patch("langchain_agentkit.extensions.hitl.tools.interrupt")
    def test_handles_non_dict_response(self, mock_interrupt):
        mock_interrupt.return_value = "unexpected"
        ext = HITLExtension(tools=True)
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

        assert "No answer provided" in result

    @patch("langchain_agentkit.extensions.hitl.tools.interrupt")
    def test_interrupt_payload_has_no_context(self, mock_interrupt):
        """ask_user questions should not include tool-approval context."""
        mock_interrupt.return_value = {"answers": {"Q?": "A"}}
        ext = HITLExtension(tools=True)
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
# InterruptConfig
# ------------------------------------------------------------------


class TestOptionPreview:
    def test_option_with_preview(self):
        opt = Option(
            label="Deploy", description="Deploy to prod", preview="```bash\ndeploy.sh\n```"
        )

        assert opt.preview == "```bash\ndeploy.sh\n```"

    def test_option_without_preview_defaults_none(self):
        opt = Option(label="Deploy", description="Deploy to prod")

        assert opt.preview is None


class TestInterruptConfig:
    def test_create_with_string_question(self):
        config = InterruptConfig(
            options=["approve", "reject"],
            question="Allow this action?",
        )

        assert config.question == "Allow this action?"
        assert config.options == ["approve", "reject"]

    def test_create_with_callable_question(self):
        config = InterruptConfig(
            options=["approve"],
            question=lambda tc: f"Allow {tc['name']}?",
        )

        assert callable(config.question)
        assert config.question({"name": "test"}) == "Allow test?"

    def test_question_defaults_to_none(self):
        config = InterruptConfig(options=["approve", "reject"])

        assert config.question is None
