# ruff: noqa: N805
"""Tests for Agent class description, tools / tools_inherit, and name metadata.

Inheritance semantics (current):
- ``tools = "inherit"`` (the default) — sub-agent borrows parent's toolset
  at delegation time.
- ``tools = []`` — explicit empty toolset; overrides any parent tools.
- ``tools = [t1, t2]`` — fixed toolset; no inheritance.
- ``tools = [t1, "inherit"]`` — sub-agent's own tools PLUS parent's tools
  (sub-agent's first; parent fills in the rest, dedupe by name).

The ``state_graph.tools_inherit`` boolean is a runtime contract the
delegation runtime reads. It's derived from the parsed ``tools`` value;
users do NOT set ``tools_inherit`` directly.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from langchain_agentkit import Agent


class TestAgentDescription:
    async def test_agent_with_description_sets_description(self):
        mock_llm = MagicMock()

        class DescribedAgent(Agent):
            model = mock_llm
            description = "A research specialist"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "described_agent"}

        graph = await DescribedAgent().graph()
        assert graph.description == "A research specialist"

    async def test_agent_without_description_sets_empty_string(self):
        mock_llm = MagicMock()

        class NoDescAgent(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "no_desc_agent"}

        graph = await NoDescAgent().graph()
        assert hasattr(graph, "description")
        assert graph.description == ""

    async def test_agent_with_empty_description_sets_empty_string(self):
        mock_llm = MagicMock()

        class EmptyDescAgent(Agent):
            model = mock_llm
            description = ""

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "empty_desc_agent"}

        graph = await EmptyDescAgent().graph()
        assert graph.description == ""


def _dummy_tool(name: str = "dummy") -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda x: x, name=name, description=f"{name} description"
    )


class TestAgentToolsInherit:
    """``tools`` is the single source of truth; ``tools_inherit`` is derived."""

    async def test_default_inherits(self):
        """Not declaring ``tools`` defaults to inherit."""
        mock_llm = MagicMock()

        class DefaultAgent(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "default_agent"}

        graph = await DefaultAgent().graph()
        assert graph.tools_inherit is True

    async def test_inherit_string_inherits(self):
        """Explicit ``tools = "inherit"`` is equivalent to the default."""
        mock_llm = MagicMock()

        class InheritAgent(Agent):
            model = mock_llm
            tools = "inherit"

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "inherit_agent"}

        graph = await InheritAgent().graph()
        assert graph.tools_inherit is True

    async def test_empty_list_overrides_inherit(self):
        """``tools = []`` is an explicit empty toolset — no inheritance."""
        mock_llm = MagicMock()

        class EmptyAgent(Agent):
            model = mock_llm
            tools = []

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "empty_agent"}

        graph = await EmptyAgent().graph()
        assert graph.tools_inherit is False

    async def test_list_without_sentinel_does_not_inherit(self):
        """A plain tool list is the full toolset — no inheritance."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        my_tool = _dummy_tool("my_tool")

        class ToolListAgent(Agent):
            model = mock_llm
            tools = [my_tool]

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "tool_list_agent"}

        graph = await ToolListAgent().graph()
        assert graph.tools_inherit is False

    async def test_list_with_inherit_sentinel_extends(self):
        """``tools = [t, "inherit"]`` adds ``t`` on top of parent's tools."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        my_tool = _dummy_tool("my_tool")

        class ExtendingAgent(Agent):
            model = mock_llm
            tools = [my_tool, "inherit"]

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "extending_agent"}

        graph = await ExtendingAgent().graph()
        assert graph.tools_inherit is True
        # The sentinel is stripped before tools reach the kit; my_tool is
        # registered on the sub-agent so the LLM can see/call it even when
        # delegation hasn't happened yet (top-level use).
        kit = graph._agentkit_kit
        tool_names = {t.name for t in kit.tools}
        assert "my_tool" in tool_names
        assert "inherit" not in tool_names

    async def test_dynamic_method_resolves_tools(self):
        """``def tools(self): return [...]`` is parsed identically."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        my_tool = _dummy_tool("my_tool")

        class DynamicAgent(Agent):
            model = mock_llm

            def tools(self):
                return [my_tool, "inherit"]

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "dynamic_agent"}

        graph = await DynamicAgent().graph()
        assert graph.tools_inherit is True
        kit = graph._agentkit_kit
        assert "my_tool" in {t.name for t in kit.tools}


class TestAgentName:
    async def test_name_defaults_to_class_name(self):
        mock_llm = MagicMock()

        class MyResearcher(Agent):
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "my_researcher"}

        graph = await MyResearcher().graph()
        assert graph.name == "MyResearcher"

    async def test_name_attribute_overrides_class_name(self):
        mock_llm = MagicMock()

        class CodeWriter(Agent):
            name = "code_writer"
            model = mock_llm

            async def handler(state, *, llm):
                return {"messages": [AIMessage(content="ok")], "sender": "code_writer"}

        graph = await CodeWriter().graph()
        assert graph.name == "code_writer"
