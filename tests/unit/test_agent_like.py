"""Tests for AgentLike composability protocol and adapters."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_agentkit.composability import AgentLike, CompiledAgent


# --- AgentLike protocol tests ---


class TestAgentLikeProtocol:
    """Test that AgentLike is a runtime-checkable protocol."""

    def test_protocol_is_runtime_checkable(self):
        class MyAgent:
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "A test agent"

            async def ainvoke(self, input: dict, config: dict | None = None) -> dict:
                return {"messages": []}

            async def astream(self, input: dict, config: dict | None = None):
                yield {"messages": []}

        agent = MyAgent()
        assert isinstance(agent, AgentLike)

    def test_non_conforming_object_fails_check(self):
        class NotAnAgent:
            pass

        assert not isinstance(NotAnAgent(), AgentLike)

    def test_partial_implementation_fails_check(self):
        class PartialAgent:
            @property
            def name(self) -> str:
                return "test"
            # Missing description, ainvoke, astream

        assert not isinstance(PartialAgent(), AgentLike)


# --- CompiledAgent adapter tests ---


class TestCompiledAgent:
    """Test CompiledAgent wraps a StateGraph correctly."""

    def test_extracts_name_from_metadata(self):
        graph = MagicMock()
        graph.agentkit_name = "researcher"
        graph.agentkit_description = "A research specialist"

        agent = CompiledAgent(graph)

        assert agent.name == "researcher"
        assert agent.description == "A research specialist"

    def test_defaults_when_no_metadata(self):
        graph = MagicMock(spec=[])  # No agentkit_* attributes

        agent = CompiledAgent(graph)

        assert agent.name == "agent"
        assert agent.description == ""

    @pytest.mark.asyncio
    async def test_ainvoke_delegates_to_graph(self):
        graph = MagicMock()
        graph.agentkit_name = "test"
        graph.agentkit_description = ""
        graph.ainvoke = AsyncMock(return_value={"messages": ["response"]})

        agent = CompiledAgent(graph)
        result = await agent.ainvoke({"messages": ["input"]})

        graph.ainvoke.assert_called_once_with({"messages": ["input"]}, None)
        assert result == {"messages": ["response"]}

    @pytest.mark.asyncio
    async def test_ainvoke_passes_config(self):
        graph = MagicMock()
        graph.agentkit_name = "test"
        graph.agentkit_description = ""
        graph.ainvoke = AsyncMock(return_value={})

        config = {"configurable": {"thread_id": "123"}}

        agent = CompiledAgent(graph)
        await agent.ainvoke({"messages": []}, config)

        graph.ainvoke.assert_called_once_with({"messages": []}, config)

    def test_is_agent_like(self):
        graph = MagicMock()
        graph.agentkit_name = "test"
        graph.agentkit_description = ""

        agent = CompiledAgent(graph)

        assert isinstance(agent, AgentLike)

    def test_graph_property_exposes_underlying_graph(self):
        graph = MagicMock()
        graph.agentkit_name = "test"
        graph.agentkit_description = ""

        agent = CompiledAgent(graph)

        assert agent.graph is graph


# --- Auto-wrapping tests ---


class TestAutoWrapping:
    """Test that raw StateGraphs can be auto-wrapped."""

    def test_wrap_if_needed_wraps_raw_graph(self):
        from langchain_agentkit.composability import wrap_if_needed

        graph = MagicMock()
        graph.agentkit_name = "researcher"
        graph.agentkit_description = "desc"

        result = wrap_if_needed(graph)

        assert isinstance(result, CompiledAgent)
        assert result.name == "researcher"

    def test_wrap_if_needed_passes_through_agent_like(self):
        from langchain_agentkit.composability import wrap_if_needed

        class MyAgent:
            @property
            def name(self):
                return "test"

            @property
            def description(self):
                return ""

            async def ainvoke(self, input, config=None):
                return {}

            async def astream(self, input, config=None):
                yield {}

        agent = MyAgent()
        result = wrap_if_needed(agent)

        assert result is agent  # Not re-wrapped

    def test_wrap_if_needed_wraps_graph_without_metadata(self):
        from langchain_agentkit.composability import wrap_if_needed

        graph = MagicMock(spec=[])  # No agentkit_* attrs

        result = wrap_if_needed(graph)

        assert isinstance(result, CompiledAgent)
        assert result.name == "agent"
