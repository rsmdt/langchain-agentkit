"""Tests for the unified Agent tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException
from langgraph.types import Command

from langchain_agentkit.tools.agent import (
    Dynamic,
    Predefined,
    _AgentDynamicInput,
    _AgentInput,
    _agent_tool,
    _build_scoped_state,
    _delegate_dynamic,
    _delegate_predefined,
    _extract_final_response,
    create_agent_tools,
)

FAKE_TOOL_CALL_ID = "call_test123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent_graph(
    name: str,
    response_content: str = "agent response",
) -> MagicMock:
    """Create a mock agent graph that returns a canned response when compiled and invoked."""
    mock_graph = MagicMock()
    mock_graph.name = name
    mock_graph.description = "Test agent"
    mock_graph.tools_inherit = False
    mock_graph.nodes = {}

    mock_compiled = AsyncMock()
    mock_compiled.ainvoke.return_value = {
        "messages": [AIMessage(content=response_content)],
        "sender": name,
    }
    mock_graph.compile.return_value = mock_compiled

    return mock_graph


# ---------------------------------------------------------------------------
# _build_scoped_state
# ---------------------------------------------------------------------------


class TestBuildScopedState:
    def test_creates_state_with_human_message(self):
        state = _build_scoped_state("do something")

        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "do something"

    def test_sets_sender_to_parent_by_default(self):
        state = _build_scoped_state("task")

        assert state["sender"] == "parent"

    def test_sets_custom_sender(self):
        state = _build_scoped_state("task", sender="lead")

        assert state["sender"] == "lead"


# ---------------------------------------------------------------------------
# _extract_final_response
# ---------------------------------------------------------------------------


class TestExtractFinalResponse:
    def test_extracts_content_from_ai_message(self):
        result = {"messages": [AIMessage(content="hello world")]}

        assert _extract_final_response(result) == "hello world"

    def test_no_messages_returns_no_response(self):
        assert _extract_final_response({"messages": []}) == "(no response)"

    def test_missing_messages_key_returns_no_response(self):
        assert _extract_final_response({}) == "(no response)"

    def test_empty_content_returns_empty_response(self):
        result = {"messages": [AIMessage(content="")]}

        assert _extract_final_response(result) == "(empty response)"


# ---------------------------------------------------------------------------
# Agent reference types
# ---------------------------------------------------------------------------


class TestAgentReferenceTypes:
    def test_predefined_has_id_field(self):
        ref = Predefined(id="researcher")

        assert ref.id == "researcher"
        assert ref.model_dump() == {"id": "researcher"}

    def test_dynamic_has_prompt_field(self):
        ref = Dynamic(prompt="You are a legal expert.")

        assert ref.prompt == "You are a legal expert."
        assert ref.model_dump() == {"prompt": "You are a legal expert."}


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class TestInputSchemas:
    def test_agent_input_only_accepts_predefined(self):
        schema = _AgentInput.model_json_schema()

        # Should not contain Dynamic/prompt variant
        schema_str = str(schema)
        assert "Predefined" in schema_str or "id" in schema_str
        assert "Dynamic" not in schema_str

    def test_agent_dynamic_input_accepts_both_variants(self):
        schema = _AgentDynamicInput.model_json_schema()

        schema_str = str(schema)
        assert "id" in schema_str
        assert "prompt" in schema_str


# ---------------------------------------------------------------------------
# _delegate_predefined
# ---------------------------------------------------------------------------


class TestDelegatePredefined:
    @pytest.mark.asyncio
    async def test_valid_agent_returns_command(self):
        agent_graph = _make_mock_agent_graph("researcher", "research result")

        result = await _delegate_predefined(
            agent_id="researcher",
            message="find info",
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "research result"
        assert result.update["messages"][0].tool_call_id == FAKE_TOOL_CALL_ID

    @pytest.mark.asyncio
    async def test_unknown_agent_raises_tool_exception(self):
        with pytest.raises(ToolException, match="not found"):
            await _delegate_predefined(
                agent_id="nonexistent",
                message="do stuff",
                agents_by_name={"researcher": MagicMock()},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
                tool_call_id=FAKE_TOOL_CALL_ID,
            )

    @pytest.mark.asyncio
    async def test_lists_available_agents_in_error(self):
        try:
            await _delegate_predefined(
                agent_id="nonexistent",
                message="do stuff",
                agents_by_name={"alpha": MagicMock(), "beta": MagicMock()},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
                tool_call_id=FAKE_TOOL_CALL_ID,
            )
        except ToolException as exc:
            assert "alpha" in str(exc)
            assert "beta" in str(exc)

    @pytest.mark.asyncio
    async def test_caches_compiled_graph(self):
        agent_graph = _make_mock_agent_graph("researcher")
        cache: dict = {}

        await _delegate_predefined(
            agent_id="researcher",
            message="task",
            agents_by_name={"researcher": agent_graph},
            compiled_cache=cache,
            delegation_timeout=30.0,
            parent_tools_getter=None,
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert "researcher" in cache

    @pytest.mark.asyncio
    async def test_exception_returns_command_with_error(self):
        agent_graph = _make_mock_agent_graph("researcher")
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.side_effect = RuntimeError("boom")
        agent_graph.compile.return_value = mock_compiled

        result = await _delegate_predefined(
            agent_id="researcher",
            message="task",
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert isinstance(result, Command)
        assert "Delegation failed" in result.update["messages"][0].content
        assert "boom" in result.update["messages"][0].content


# ---------------------------------------------------------------------------
# _delegate_dynamic
# ---------------------------------------------------------------------------


class TestDelegateDynamic:
    @pytest.mark.asyncio
    async def test_empty_prompt_raises_tool_exception(self):
        with pytest.raises(ToolException, match="prompt cannot be empty"):
            await _delegate_dynamic(
                prompt="",
                message="do something",
                delegation_timeout=30.0,
                parent_llm_getter=lambda: MagicMock(),
                tool_call_id=FAKE_TOOL_CALL_ID,
            )

    @pytest.mark.asyncio
    async def test_whitespace_prompt_raises(self):
        with pytest.raises(ToolException, match="prompt cannot be empty"):
            await _delegate_dynamic(
                prompt="   ",
                message="do something",
                delegation_timeout=30.0,
                parent_llm_getter=lambda: MagicMock(),
                tool_call_id=FAKE_TOOL_CALL_ID,
            )

    @pytest.mark.asyncio
    async def test_valid_prompt_returns_command(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="analysis result")

        result = await _delegate_dynamic(
            prompt="You are a data analyst.",
            message="analyze this",
            delegation_timeout=30.0,
            parent_llm_getter=lambda: mock_llm,
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "analysis result"

    @pytest.mark.asyncio
    async def test_none_llm_getter_raises(self):
        with pytest.raises(ToolException, match="parent LLM"):
            await _delegate_dynamic(
                prompt="You are an expert.",
                message="do something",
                delegation_timeout=30.0,
                parent_llm_getter=None,
                tool_call_id=FAKE_TOOL_CALL_ID,
            )


# ---------------------------------------------------------------------------
# _agent_tool (unified entry point)
# ---------------------------------------------------------------------------


class TestAgentTool:
    @pytest.mark.asyncio
    async def test_routes_predefined_agent_by_dict(self):
        agent_graph = _make_mock_agent_graph("researcher", "result")

        result = await _agent_tool(
            agent={"id": "researcher"},
            message="find info",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "result"

    @pytest.mark.asyncio
    async def test_routes_predefined_agent_by_model(self):
        agent_graph = _make_mock_agent_graph("researcher", "result")

        result = await _agent_tool(
            agent=Predefined(id="researcher"),
            message="find info",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "result"

    @pytest.mark.asyncio
    async def test_routes_dynamic_agent(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="dynamic result")

        result = await _agent_tool(
            agent={"prompt": "You are an expert."},
            message="analyze this",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=True,
            parent_llm_getter=lambda: mock_llm,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "dynamic result"

    @pytest.mark.asyncio
    async def test_dynamic_agent_when_ephemeral_disabled_raises(self):
        with pytest.raises(ToolException, match="not enabled"):
            await _agent_tool(
                agent={"prompt": "You are an expert."},
                message="analyze this",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                agents_by_name={},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
                ephemeral=False,
                parent_llm_getter=None,
            )

    @pytest.mark.asyncio
    async def test_invalid_agent_ref_raises(self):
        with pytest.raises(ToolException, match="Invalid agent reference"):
            await _agent_tool(
                agent={"unknown_field": "value"},
                message="do stuff",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                agents_by_name={},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
                ephemeral=False,
                parent_llm_getter=None,
            )


# ---------------------------------------------------------------------------
# create_agent_tools
# ---------------------------------------------------------------------------


class TestCreateAgentTools:
    def test_returns_single_agent_tool(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert len(tools) == 1
        assert tools[0].name == "Agent"

    def test_returns_single_tool_with_ephemeral(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=True,
            parent_llm_getter=lambda: MagicMock(),
        )

        assert len(tools) == 1
        assert tools[0].name == "Agent"

    def test_ephemeral_without_llm_getter_raises_value_error(self):
        with pytest.raises(ValueError, match="parent_llm_getter is required"):
            create_agent_tools(
                agents_by_name={"researcher": MagicMock()},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
                ephemeral=True,
                parent_llm_getter=None,
            )

    def test_tool_has_description(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert tools[0].description

    def test_schema_without_ephemeral_uses_agent_input(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert tools[0].args_schema is _AgentInput

    def test_schema_with_ephemeral_uses_dynamic_input(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=True,
            parent_llm_getter=lambda: MagicMock(),
        )

        assert tools[0].args_schema is _AgentDynamicInput
