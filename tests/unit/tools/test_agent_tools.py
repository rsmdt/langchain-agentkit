"""Tests for Delegate and DelegateEphemeral tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException
from langgraph.types import Command

from langchain_agentkit.tools.agent import (
    _build_scoped_state,
    _delegate,
    _delegate_ephemeral,
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
    mock_graph.agentkit_name = name
    mock_graph.agentkit_description = "Test agent"
    mock_graph.agentkit_tools_inherit = False
    mock_graph.nodes = {}

    # compiled.ainvoke returns a canned result
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
# _delegate
# ---------------------------------------------------------------------------


class TestDelegate:
    @pytest.mark.asyncio
    async def test_delegate_with_valid_agent_returns_command(self):
        agent_graph = _make_mock_agent_graph("researcher", "research result")

        result = await _delegate(
            agent="researcher",
            message="find info",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "research result"
        assert result.update["messages"][0].tool_call_id == FAKE_TOOL_CALL_ID

    @pytest.mark.asyncio
    async def test_delegate_with_unknown_agent_raises_tool_exception(self):
        with pytest.raises(ToolException, match="not found"):
            await _delegate(
                agent="nonexistent",
                message="do stuff",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                agents_by_name={"researcher": MagicMock()},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
            )

    @pytest.mark.asyncio
    async def test_delegate_lists_available_agents_in_error(self):
        try:
            await _delegate(
                agent="nonexistent",
                message="do stuff",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                agents_by_name={"alpha": MagicMock(), "beta": MagicMock()},
                compiled_cache={},
                delegation_timeout=30.0,
                parent_tools_getter=None,
            )
        except ToolException as exc:
            assert "alpha" in str(exc)
            assert "beta" in str(exc)

    @pytest.mark.asyncio
    async def test_delegate_caches_compiled_graph(self):
        agent_graph = _make_mock_agent_graph("researcher")
        cache: dict = {}

        await _delegate(
            agent="researcher",
            message="task",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={"researcher": agent_graph},
            compiled_cache=cache,
            delegation_timeout=30.0,
            parent_tools_getter=None,
        )

        assert "researcher" in cache

    @pytest.mark.asyncio
    async def test_delegate_exception_returns_command_with_error(self):
        agent_graph = _make_mock_agent_graph("researcher")
        mock_compiled = AsyncMock()
        mock_compiled.ainvoke.side_effect = RuntimeError("boom")
        agent_graph.compile.return_value = mock_compiled

        result = await _delegate(
            agent="researcher",
            message="task",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            agents_by_name={"researcher": agent_graph},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
        )

        assert isinstance(result, Command)
        assert "Delegation failed" in result.update["messages"][0].content
        assert "boom" in result.update["messages"][0].content


# ---------------------------------------------------------------------------
# _delegate_ephemeral
# ---------------------------------------------------------------------------


class TestDelegateEphemeral:
    @pytest.mark.asyncio
    async def test_ephemeral_with_empty_instructions_raises_tool_exception(self):
        mock_llm = MagicMock()

        with pytest.raises(ToolException, match="instructions cannot be empty"):
            await _delegate_ephemeral(
                message="do something",
                instructions="",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                delegation_timeout=30.0,
                parent_llm_getter=lambda: mock_llm,
            )

    @pytest.mark.asyncio
    async def test_ephemeral_with_whitespace_instructions_raises(self):
        mock_llm = MagicMock()

        with pytest.raises(ToolException, match="instructions cannot be empty"):
            await _delegate_ephemeral(
                message="do something",
                instructions="   ",
                state={"messages": []},
                tool_call_id=FAKE_TOOL_CALL_ID,
                delegation_timeout=30.0,
                parent_llm_getter=lambda: mock_llm,
            )

    @pytest.mark.asyncio
    async def test_ephemeral_with_valid_instructions_returns_command(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="analysis result")

        result = await _delegate_ephemeral(
            message="analyze this",
            instructions="You are a data analyst.",
            state={"messages": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
            delegation_timeout=30.0,
            parent_llm_getter=lambda: mock_llm,
        )

        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "analysis result"


# ---------------------------------------------------------------------------
# create_agent_tools
# ---------------------------------------------------------------------------


class TestCreateAgentTools:
    def test_returns_delegate_only_without_ephemeral(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=False,
            parent_llm_getter=None,
        )

        assert len(tools) == 1
        assert tools[0].name == "Delegate"

    def test_returns_delegate_and_ephemeral_when_enabled(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=True,
            parent_llm_getter=lambda: MagicMock(),
        )

        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "Delegate" in names
        assert "DelegateEphemeral" in names

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

    def test_tools_have_descriptions(self):
        tools = create_agent_tools(
            agents_by_name={"researcher": MagicMock()},
            compiled_cache={},
            delegation_timeout=30.0,
            parent_tools_getter=None,
            ephemeral=True,
            parent_llm_getter=lambda: MagicMock(),
        )

        for tool in tools:
            assert tool.description, f"{tool.name} has no description"
