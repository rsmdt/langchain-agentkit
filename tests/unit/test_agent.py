# ruff: noqa: N805
"""Tests for the Agent base class."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.agent import Agent


class TestAgentClass:
    """Tests for the Agent base class with flexible property resolution."""

    async def test_graph_returns_state_graph(self):
        """Agent.graph() returns an uncompiled StateGraph."""
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class Static(Agent):
            model = mock_llm
            prompt = "You are helpful."

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "static"}

        result = await Static().graph()

        assert isinstance(result, StateGraph)

    async def test_compile_returns_runnable(self):
        """Agent.compile() returns a compiled, invocable graph."""
        mock_llm = MagicMock()

        class Static(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "static"}

        compiled = await Static().compile()

        assert hasattr(compiled, "invoke")
        assert hasattr(compiled, "ainvoke")

    def test_kwargs_stored_as_instance_attributes(self):
        """__init__(**kwargs) stores values as instance attributes."""
        mock_llm = MagicMock()

        class WithBackend(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        instance = WithBackend(backend="my-backend", config={"key": "val"})

        assert instance.backend == "my-backend"
        assert instance.config == {"key": "val"}

    async def test_sync_method_resolved(self):
        """Sync methods are called during graph()."""
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class SyncMethod(Agent):
            model = mock_llm

            def prompt(self):
                return "Dynamic prompt"

            async def handler(state, *, prompt):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        result = await SyncMethod().graph()

        assert isinstance(result, StateGraph)

    async def test_async_method_resolved(self):
        """Async methods are awaited during graph()."""
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class AsyncMethod(Agent):
            model = mock_llm

            async def prompt(self):
                return "Async prompt"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        result = await AsyncMethod().graph()

        assert isinstance(result, StateGraph)

    async def test_model_resolver_not_called_by_resolve(self):
        """model_resolver is returned as-is, not called with zero args."""
        resolver_fn = MagicMock(return_value=MagicMock())

        class WithResolver(Agent):
            model = "gpt-4o"
            model_resolver = staticmethod(resolver_fn)

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        await WithResolver().compile()

        # resolver_fn should have been called by AgentKit to resolve "gpt-4o",
        # not by _resolve() with zero args
        resolver_fn.assert_called_once_with("gpt-4o")

    async def test_mixed_static_and_dynamic(self):
        """Mix of static attributes and dynamic methods."""
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class Mixed(Agent):
            model = mock_llm
            tools = []

            def extensions(self):
                return []

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        result = await Mixed().graph()

        assert isinstance(result, StateGraph)

    async def test_kwargs_accessible_in_methods(self):
        """Instance kwargs from __init__ are accessible via self in methods."""
        from langgraph.graph import StateGraph

        mock_llm = MagicMock()

        class Dynamic(Agent):
            model = mock_llm

            def prompt(self):
                return f"Backend: {self.backend_name}"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        instance = Dynamic(backend_name="test-backend")
        result = await instance.graph()

        assert isinstance(result, StateGraph)

    async def test_compiles_and_invokes(self):
        """Full end-to-end: compile() returns a runnable."""
        mock_llm = MagicMock()

        class Simple(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="agent done")], "sender": "simple"}

        compiled = await Simple().compile()
        result = await compiled.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert result["messages"][-1].content == "agent done"

    async def test_compile_forwards_kwargs(self):
        """compile(**kwargs) forwards to StateGraph.compile()."""
        from langgraph.checkpoint.memory import InMemorySaver

        mock_llm = MagicMock()

        class WithCheckpointer(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        compiled = await WithCheckpointer().compile(checkpointer=InMemorySaver())

        assert hasattr(compiled, "ainvoke")

    async def test_instance_attribute_overrides_class_attribute(self):
        """kwargs override class-level defaults."""
        from langgraph.graph import StateGraph

        mock_llm_class = MagicMock()
        mock_llm_instance = MagicMock()

        class Override(Agent):
            model = mock_llm_class

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "t"}

        result = await Override(model=mock_llm_instance).graph()

        assert isinstance(result, StateGraph)

    async def test_compile_applies_max_turns_as_recursion_limit(self):
        """Agent.max_turns is translated to recursion_limit = max_turns * 2."""
        mock_llm = MagicMock()

        class Bounded(Agent):
            model = mock_llm
            max_turns = 7

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "b"}

        compiled = await Bounded().compile()

        assert compiled.config["recursion_limit"] == 14

    async def test_compile_explicit_recursion_limit_wins(self):
        """Caller-supplied recursion_limit overrides max_turns translation."""
        mock_llm = MagicMock()

        class Bounded(Agent):
            model = mock_llm
            max_turns = 7

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "b"}

        compiled = await Bounded().compile(recursion_limit=99)

        assert compiled.config["recursion_limit"] == 99

    async def test_compile_no_max_turns_omits_recursion_limit(self):
        """When max_turns is unset, no recursion_limit override is applied."""
        mock_llm = MagicMock()

        class Unbounded(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "u"}

        compiled = await Unbounded().compile()

        # No explicit recursion_limit in the bound config (LangGraph applies
        # its default of 25 at invocation if absent). ``config`` itself may
        # be ``None`` when nothing was bound via ``with_config``.
        assert "recursion_limit" not in (compiled.config or {})
