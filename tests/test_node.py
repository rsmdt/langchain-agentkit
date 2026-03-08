# ruff: noqa: N801, N805
"""Tests for the node metaclass."""

from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from langchain_agentkit.node import (
    _normalize_skills,
    _validate_handler_signature,
)
from langchain_agentkit.skill_registry import SkillRegistry
from langchain_agentkit.state import AgentState

FIXTURES = Path(__file__).parent / "fixtures"


class TestValidateHandlerSignature:
    def test_accepts_state_only(self):
        def handler(state):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == set()
        assert state_type is AgentState

    def test_accepts_state_with_llm(self):
        def handler(state, *, llm):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}
        assert state_type is AgentState

    def test_accepts_all_injectables(self):
        def handler(state, *, llm, tools, runtime):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm", "tools", "runtime"}

    def test_rejects_empty_signature(self):
        def handler():
            pass

        with pytest.raises(ValueError, match="at least"):
            _validate_handler_signature(handler, "test")

    def test_rejects_unknown_keyword_param(self):
        def handler(state, *, unknown):
            pass

        with pytest.raises(ValueError, match="unknown handler parameter"):
            _validate_handler_signature(handler, "test")

    def test_rejects_positional_after_state(self):
        def handler(state, extra):
            pass

        with pytest.raises(ValueError, match="keyword-only"):
            _validate_handler_signature(handler, "test")

    def test_accepts_kwargs(self):
        def handler(state, *, llm, **kwargs):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}

    def test_extracts_state_type_from_annotation(self):
        class CustomState(TypedDict, total=False):
            messages: Annotated[list[Any], add_messages]
            draft: dict | None

        def handler(state: CustomState, *, llm):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}
        assert state_type is CustomState

    def test_defaults_to_agent_state_without_annotation(self):
        def handler(state, *, llm):
            pass

        _, state_type = _validate_handler_signature(handler, "test")

        assert state_type is AgentState


class TestNormalizeSkills:
    def test_none_returns_none(self):
        assert _normalize_skills(None) is None

    def test_string_returns_skill_registry(self):
        result = _normalize_skills(str(FIXTURES / "skills"))

        assert isinstance(result, SkillRegistry)

    def test_list_returns_skill_registry(self):
        result = _normalize_skills([str(FIXTURES / "skills")])

        assert isinstance(result, SkillRegistry)

    def test_skill_registry_passes_through(self):
        kit = SkillRegistry(str(FIXTURES / "skills"))

        result = _normalize_skills(kit)

        assert result is kit

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_skills(123)


class TestNodeMetaclass:
    def test_returns_uncompiled_state_graph(self):
        from langgraph.graph import StateGraph

        from langchain_agentkit import node

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class test_agent(node):
            llm = mock_llm
            tools = []

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="hello")],
                    "sender": "test_agent",
                }

        # The metaclass returns a StateGraph, not a CompiledStateGraph
        assert isinstance(test_agent, StateGraph)
        assert hasattr(test_agent, "compile")
        assert not hasattr(test_agent, "invoke")

    def test_requires_handler(self):
        from langchain_agentkit import node

        with pytest.raises(ValueError, match="must define.*handler"):

            class bad_agent(node):
                llm = MagicMock()

    def test_requires_llm(self):
        from langchain_agentkit import node

        with pytest.raises(ValueError, match="must define.*llm"):

            class bad_agent(node):
                async def handler(state):
                    return {"messages": [], "sender": "bad"}

    def test_handler_must_be_callable(self):
        from langchain_agentkit import node

        with pytest.raises(ValueError, match="callable"):

            class bad_agent(node):
                llm = MagicMock()
                handler = "not a function"

    def test_tools_must_be_list(self):
        from langchain_agentkit import node

        with pytest.raises(ValueError, match="tools must be a list"):

            class bad_agent(node):
                llm = MagicMock()
                tools = "not a list"

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}

    def test_node_with_skills(self):
        from langgraph.graph import StateGraph

        from langchain_agentkit import node

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class skilled_agent(node):
            llm = mock_llm
            skills = str(FIXTURES / "skills")

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="skilled")],
                    "sender": "skilled_agent",
                }

        assert isinstance(skilled_agent, StateGraph)

    @pytest.mark.asyncio
    async def test_custom_state_type_from_handler_annotation(self):
        from langchain_agentkit import node

        class WorkflowState(TypedDict, total=False):
            messages: Annotated[list[Any], add_messages]
            draft: dict | None

        mock_llm = MagicMock()

        class custom_agent(node):
            llm = mock_llm

            async def handler(state: WorkflowState, *, llm):
                return {"messages": [AIMessage(content="custom")]}

        # The StateGraph should use WorkflowState, not AgentState
        compiled = custom_agent.compile()
        result = await compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "draft": {"section": "content"},
            }
        )

        # Custom state fields survive graph execution
        assert result["draft"] == {"section": "content"}

    @pytest.mark.asyncio
    async def test_default_state_type_without_annotation(self):
        from langchain_agentkit import node

        mock_llm = MagicMock()

        class default_agent(node):
            llm = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="default")]}

        # Should compile and work with AgentState (default)
        compiled = default_agent.compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert result["messages"][-1].content == "default"


class TestNodeInvocation:
    @pytest.mark.asyncio
    async def test_no_tools_node_invokes_handler(self):
        from langchain_agentkit import node

        mock_llm = MagicMock()

        class simple(node):
            llm = mock_llm

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="done")],
                    "sender": "simple",
                }

        compiled = simple.compile()
        result = await compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
            }
        )

        assert result["messages"][-1].content == "done"
