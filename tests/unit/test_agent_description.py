# ruff: noqa: N801
"""Tests for agent metaclass description, tools='inherit', and agentkit_name."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from langchain_agentkit.agent import agent


class TestAgentDescription:
    def test_agent_with_description_sets_agentkit_description(self):
        mock_llm = MagicMock()

        class described_agent(agent):
            llm = mock_llm
            description = "A research specialist"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "described_agent"}

        assert isinstance(described_agent, StateGraph)
        assert described_agent.agentkit_description == "A research specialist"

    def test_agent_without_description_sets_empty_string(self):
        mock_llm = MagicMock()

        class no_desc_agent(agent):
            llm = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "no_desc_agent"}

        assert hasattr(no_desc_agent, "agentkit_description")
        assert no_desc_agent.agentkit_description == ""

    def test_agent_with_empty_description_sets_empty_string(self):
        mock_llm = MagicMock()

        class empty_desc_agent(agent):
            llm = mock_llm
            description = ""

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "empty_desc_agent"}

        assert empty_desc_agent.agentkit_description == ""


class TestAgentToolsInherit:
    def test_tools_inherit_string_sets_flag_true(self):
        mock_llm = MagicMock()

        class inheriting_agent(agent):
            llm = mock_llm
            tools = "inherit"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "inheriting_agent"}

        assert inheriting_agent.agentkit_tools_inherit is True

    def test_tools_list_sets_flag_false(self):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        from langchain_core.tools import StructuredTool

        dummy_tool = StructuredTool.from_function(
            func=lambda x: x, name="dummy", description="dummy"
        )

        class list_tools_agent(agent):
            llm = mock_llm
            tools = [dummy_tool]

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "list_tools_agent"}

        assert list_tools_agent.agentkit_tools_inherit is False

    def test_no_tools_sets_flag_false(self):
        mock_llm = MagicMock()

        class no_tools_agent(agent):
            llm = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "no_tools_agent"}

        assert no_tools_agent.agentkit_tools_inherit is False

    def test_tools_invalid_string_raises_value_error(self):
        mock_llm = MagicMock()

        with pytest.raises(ValueError, match="tools must be a list or 'inherit'"):

            class bad_tools_agent(agent):
                llm = mock_llm
                tools = "invalid"

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}

    def test_empty_tools_list_sets_flag_false(self):
        mock_llm = MagicMock()

        class empty_tools_agent(agent):
            llm = mock_llm
            tools = []

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "empty_tools_agent"}

        assert empty_tools_agent.agentkit_tools_inherit is False


class TestAgentKitName:
    def test_agentkit_name_set_to_class_name(self):
        mock_llm = MagicMock()

        class my_researcher(agent):
            llm = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "my_researcher"}

        assert my_researcher.agentkit_name == "my_researcher"

    def test_agentkit_name_reflects_different_class_names(self):
        mock_llm = MagicMock()

        class code_writer(agent):
            llm = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "code_writer"}

        assert code_writer.agentkit_name == "code_writer"
