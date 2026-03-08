# ruff: noqa: N801, N805
"""Tests for the agent metaclass."""

from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph.message import add_messages

from langchain_agentkit.agent import (
    _validate_handler_signature,
    agent,
)
from langchain_agentkit.state import AgentState

FIXTURES = Path(__file__).parent / "fixtures"


class TestValidateHandlerSignature:
    def test_accepts_state_only(self):
        def handler(state):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == set()
        assert state_type is AgentState

    def test_accepts_all_injectables(self):
        def handler(state, *, llm, tools, prompt, config, runtime):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm", "tools", "prompt", "config", "runtime"}

    def test_rejects_unknown_keyword_param(self):
        def handler(state, *, unknown):
            pass

        with pytest.raises(ValueError, match="unknown handler parameter"):
            _validate_handler_signature(handler, "test")

    def test_rejects_empty_signature(self):
        def handler():
            pass

        with pytest.raises(ValueError, match="at least"):
            _validate_handler_signature(handler, "test")

    def test_extracts_state_type_from_annotation(self):
        class CustomState(TypedDict, total=False):
            messages: Annotated[list[Any], add_messages]
            draft: dict | None

        def handler(state: CustomState, *, llm):
            pass

        injectable, state_type = _validate_handler_signature(handler, "test")

        assert injectable == {"llm"}
        assert state_type is CustomState

    def test_defaults_to_agent_state(self):
        def handler(state, *, llm):
            pass

        _, state_type = _validate_handler_signature(handler, "test")

        assert state_type is AgentState

    def test_rejects_positional_param_after_state(self):
        """Positional params after state must be keyword-only."""

        def handler(state, extra):
            pass

        with pytest.raises(ValueError, match="keyword-only"):
            _validate_handler_signature(handler, "test")


class TestAgentMetaclass:
    def test_returns_uncompiled_state_graph(self):
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class test_agent(agent):
            llm = mock_llm
            tools = []

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="hello")],
                    "sender": "test_agent",
                }

        assert isinstance(test_agent, StateGraph)
        assert hasattr(test_agent, "compile")
        assert not hasattr(test_agent, "invoke")

    def test_requires_handler(self):
        with pytest.raises(ValueError, match="must define.*handler"):

            class bad_agent(agent):
                llm = MagicMock()

    def test_requires_llm(self):
        with pytest.raises(ValueError, match="must define.*llm"):

            class bad_agent(agent):
                async def handler(state):
                    return {"messages": [], "sender": "bad"}

    def test_handler_must_be_callable(self):
        with pytest.raises(ValueError, match="callable"):

            class bad_agent(agent):
                llm = MagicMock()
                handler = "not a function"

    def test_tools_must_be_list(self):
        with pytest.raises(ValueError, match="tools must be a list"):

            class bad_agent(agent):
                llm = MagicMock()
                tools = "not a list"

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}

    def test_middleware_must_be_list(self):
        with pytest.raises(ValueError, match="middleware must be a list"):

            class bad_agent(agent):
                llm = MagicMock()
                middleware = "not a list"

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}

    def test_agent_with_middleware_produces_state_graph(self):
        from langgraph.graph import StateGraph

        from langchain_agentkit.skills_middleware import SkillsMiddleware

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class skilled_agent(agent):
            llm = mock_llm
            middleware = [SkillsMiddleware(str(FIXTURES / "skills"))]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="skilled")],
                    "sender": "skilled_agent",
                }

        assert isinstance(skilled_agent, StateGraph)

    def test_agent_with_prompt_template_produces_state_graph(self):
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class prompted_agent(agent):
            llm = mock_llm
            prompt = "You are a helpful assistant."

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="ok")],
                    "sender": "prompted_agent",
                }

        assert isinstance(prompted_agent, StateGraph)


class TestAgentInvocation:
    @pytest.mark.asyncio
    async def test_no_tools_agent_invokes_handler(self):
        mock_llm = MagicMock()

        class simple(agent):
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

    @pytest.mark.asyncio
    async def test_agent_with_prompt_injects_composed_prompt(self):
        mock_llm = MagicMock()
        captured = {}

        class prompt_agent(agent):
            llm = mock_llm
            prompt = "You are helpful."

            async def handler(state, *, llm, prompt):
                captured["prompt"] = prompt
                return {"messages": [AIMessage(content="ok")], "sender": "prompt_agent"}

        compiled = prompt_agent.compile()
        await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert captured["prompt"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_agent_with_middleware_injects_composed_prompt(self):
        """Verify middleware prompt sections are composed and injected into handler."""
        mock_llm = MagicMock()
        captured = {}

        class StubMW:
            @property
            def tools(self):
                return []

            def prompt(self, state, config):
                return "Middleware section"

        class mw_agent(agent):
            llm = mock_llm
            middleware = [StubMW()]
            prompt = "Base template"

            async def handler(state, *, prompt):
                captured["prompt"] = prompt
                return {"messages": [AIMessage(content="ok")], "sender": "mw_agent"}

        compiled = mw_agent.compile()
        await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert "Base template" in captured["prompt"]
        assert "Middleware section" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_agent_with_middleware_tools_injected(self):
        """Verify middleware tools are merged with user tools and injected."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        captured = {}

        mw_tool = StructuredTool.from_function(
            func=lambda x: x, name="mw_tool", description="from middleware"
        )

        class ToolMW:
            @property
            def tools(self):
                return [mw_tool]

            def prompt(self, state, config):
                return None

        user_tool = StructuredTool.from_function(
            func=lambda x: x, name="user_tool", description="from user"
        )

        class tools_agent(agent):
            llm = mock_llm
            tools = [user_tool]
            middleware = [ToolMW()]

            async def handler(state, *, tools):
                captured["tools"] = tools
                return {"messages": [AIMessage(content="ok")], "sender": "tools_agent"}

        compiled = tools_agent.compile()
        await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        tool_names = [t.name for t in captured["tools"]]
        assert "user_tool" in tool_names
        assert "mw_tool" in tool_names

    @pytest.mark.asyncio
    async def test_react_loop_routes_tool_calls(self):
        """Verify tool calls route through ToolNode and back to handler."""
        call_count = {"n": 0}

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        mock_llm = MagicMock()

        class react_agent(agent):
            llm = mock_llm
            tools = [greet]

            async def handler(state, *, llm):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": "greet", "args": {"name": "World"}, "id": "1"}
                                ],
                            )
                        ],
                        "sender": "react_agent",
                    }
                return {
                    "messages": [AIMessage(content="Done greeting")],
                    "sender": "react_agent",
                }

        compiled = react_agent.compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert call_count["n"] == 2
        messages = result["messages"]
        assert len(messages) == 4
        assert messages[-1].content == "Done greeting"

    @pytest.mark.asyncio
    async def test_sync_handler_works(self):
        """Verify sync handlers work via isawaitable check."""
        mock_llm = MagicMock()

        class sync_agent(agent):
            llm = mock_llm

            def handler(state, *, llm):  # noqa: N805
                return {
                    "messages": [AIMessage(content="sync done")],
                    "sender": "sync_agent",
                }

        compiled = sync_agent.compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert result["messages"][-1].content == "sync done"

    def test_hitl_middleware_wrap_tool_call_passed_to_tool_node(self):
        """Verify agent detects wrap_tool_call from HITLMiddleware."""
        from langchain_agentkit.hitl_middleware import HITLMiddleware

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        @tool
        def send_email(to: str, body: str) -> str:
            """Send an email."""
            return f"Sent to {to}"

        hitl = HITLMiddleware(interrupt_on={"send_email": True})

        class hitl_agent(agent):
            llm = mock_llm
            tools = [send_email]
            middleware = [hitl]

            async def handler(state, *, llm):
                return {
                    "messages": [AIMessage(content="ok")],
                    "sender": "hitl_agent",
                }

        # Should produce a StateGraph without error
        from langgraph.graph import StateGraph

        assert isinstance(hitl_agent, StateGraph)

    def test_multiple_wrap_tool_call_raises(self):
        """Only one middleware can provide wrap_tool_call."""
        from langchain_agentkit.hitl_middleware import HITLMiddleware

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        @tool
        def action(x: str) -> str:
            """Do something."""
            return x

        with pytest.raises(ValueError, match="multiple middleware provide wrap_tool_call"):

            class bad_agent(agent):
                llm = mock_llm
                tools = [action]
                middleware = [
                    HITLMiddleware(interrupt_on={"action": True}),
                    HITLMiddleware(interrupt_on={"action": True}),
                ]

                async def handler(state, *, llm):
                    return {"messages": [], "sender": "bad"}
