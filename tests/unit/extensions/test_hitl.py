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
    def test_bool_true_expands_to_approve_reject(self):
        ext = HITLExtension(interrupt_on={"send_email": True})

        assert "send_email" in ext.interrupt_on
        assert ext.interrupt_on["send_email"].options == [
            "approve",
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

    def test_state_schema_returns_none(self):
        ext = HITLExtension()

        assert ext.state_schema is None

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
# wrap_tool — auto-approved tools
# ------------------------------------------------------------------


class TestWrapToolAutoApproved:
    @pytest.mark.asyncio
    async def test_unconfigured_tool_executes_normally(self):
        ext = HITLExtension(interrupt_on={"send_email": True})
        mock_request = MagicMock()
        mock_request.tool_call = {
            "name": "search",
            "args": {"q": "test"},
            "id": "call_1",
        }
        expected = ToolMessage(content="result", tool_call_id="call_1")
        mock_handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(
            state=mock_request, handler=mock_handler, runtime=_TEST_RUNTIME
        )

        mock_handler.assert_called_once_with(mock_request)
        assert result == expected

    @pytest.mark.asyncio
    async def test_single_approve_decision_auto_executes(self):
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
        mock_handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(
            state=mock_request, handler=mock_handler, runtime=_TEST_RUNTIME
        )

        mock_handler.assert_called_once_with(mock_request)
        assert result == expected

    @pytest.mark.asyncio
    async def test_single_reject_decision_auto_rejects(self):
        """Only one allowed decision (reject) — auto-reject without asking."""
        ext = HITLExtension(
            interrupt_on={
                "search": InterruptConfig(options=["reject"]),
            },
        )
        mock_request = MagicMock()
        mock_request.tool_call = {
            "name": "search",
            "args": {"q": "test"},
            "id": "call_1",
        }
        mock_handler = AsyncMock()

        result = await ext.wrap_tool(
            state=mock_request, handler=mock_handler, runtime=_TEST_RUNTIME
        )

        mock_handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "Auto-rejected search (only allowed option: reject)"


# ------------------------------------------------------------------
# wrap_tool — interrupt flow
# ------------------------------------------------------------------


class TestWrapToolInterrupt:
    def _make_request(self, tool_name="send_email", args=None):
        mock = MagicMock()
        mock.tool_call = {
            "name": tool_name,
            "args": args or {"to": "a@b.com"},
            "id": "call_1",
        }
        return mock

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_interrupt_payload_uses_question_format(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "question"
        assert len(payload["questions"]) == 1
        q = payload["questions"][0]
        assert "question" in q
        assert "header" in q
        assert "options" in q
        assert q["context"]["tool"] == "send_email"
        assert q["context"]["args"] == {"to": "a@b.com"}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_default_options_are_approve_reject_only(self, mock_interrupt):
        """interrupt.value shape is unchanged except the default options are
        exactly [Approve, Reject] — no Edit."""
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "question"
        assert len(payload["questions"]) == 1
        q = payload["questions"][0]
        assert [o["label"] for o in q["options"]] == ["Approve", "Reject"]
        assert q["context"] == {"tool": "send_email", "args": {"to": "a@b.com"}}

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_options_match_options(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": {"options": ["approve", "reject"]},
            },
        )
        request = self._make_request()

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        q = mock_interrupt.call_args[0][0]["questions"][0]
        labels = [o["label"] for o in q["options"]]
        assert labels == ["Approve", "Reject"]

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_approve_executes_tool(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        expected = ToolMessage(content="sent", tool_call_id="call_1")
        mock_handler = AsyncMock(return_value=expected)

        result = await ext.wrap_tool(state=request, handler=mock_handler, runtime=_TEST_RUNTIME)

        mock_handler.assert_called_once_with(request)
        assert result == expected

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_reject_returns_fixed_error(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Reject"}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=mock_handler, runtime=_TEST_RUNTIME)

        mock_handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "User rejected the send_email tool call."

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_free_form_answer_punts_to_llm(self, mock_interrupt):
        """Anything other than Approve/Reject is forwarded to the LLM verbatim."""
        mock_interrupt.return_value = {
            "answers": {"0": "use path deliverables/h2-sizing.md"},
        }
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=mock_handler, runtime=_TEST_RUNTIME)

        mock_handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert (
            result.content
            == "User responded instead of approving: use path deliverables/h2-sizing.md"
        )

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_edited_args_on_resume_is_ignored(self, mock_interrupt):
        """edited_args is no longer read — a non-Approve/Reject answer still
        punts to the LLM and the tool never runs."""
        mock_interrupt.return_value = {
            "answers": {"0": "Edit"},
            "edited_args": {"to": "new@b.com"},
        }
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=mock_handler, runtime=_TEST_RUNTIME)

        mock_handler.assert_not_called()
        request.override.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "User responded instead of approving: Edit"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_empty_response_punts_with_no_response(self, mock_interrupt):
        mock_interrupt.return_value = {}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        result = await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "User responded instead of approving: (no response)"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_non_string_answer_punts_with_no_response(self, mock_interrupt):
        """A non-string value at index 0 (e.g. a list) is treated as no response."""
        mock_interrupt.return_value = {"answers": {"0": ["a", "b"]}}
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()
        mock_handler = AsyncMock()

        result = await ext.wrap_tool(state=request, handler=mock_handler, runtime=_TEST_RUNTIME)

        mock_handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.content == "User responded instead of approving: (no response)"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_non_dict_response_returns_error(self, mock_interrupt):
        mock_interrupt.return_value = "unexpected string"
        ext = HITLExtension(interrupt_on={"send_email": True})
        request = self._make_request()

        result = await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_custom_description_string(self, mock_interrupt):
        mock_interrupt.return_value = {"answers": {"0": "Approve"}}
        ext = HITLExtension(
            interrupt_on={
                "send_email": InterruptConfig(
                    options=["approve", "reject"],
                    question="Send email to user?",
                ),
            },
        )
        request = self._make_request()

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["question"] == "Send email to user?"

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
        request = self._make_request()

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert q["question"] == "Email a@b.com?"

    @pytest.mark.asyncio
    @patch("langchain_agentkit.extensions.hitl.extension.interrupt")
    async def test_header_truncated_to_12_chars(self, mock_interrupt):
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

        await ext.wrap_tool(state=request, handler=AsyncMock(), runtime=_TEST_RUNTIME)

        q = mock_interrupt.call_args[0][0]["questions"][0]
        assert len(q["header"]) <= 12


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
