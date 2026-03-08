"""Tests for Middleware protocol and AgentKit composition engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from langchain_agentkit.agent_kit import AgentKit

if TYPE_CHECKING:
    from langchain_agentkit.middleware import Middleware

FIXTURES = Path(__file__).parent / "fixtures"


def _make_tool(name: str, description: str = "stub") -> StructuredTool:
    def _fn(x: str) -> str:
        return x

    return StructuredTool.from_function(func=_fn, name=name, description=description)


class StubMiddleware:
    def __init__(self, tools=None, prompt_text=None):
        self._tools = tools or []
        self._prompt_text = prompt_text

    @property
    def tools(self):
        return self._tools

    def prompt(self, state, config):
        return self._prompt_text


class TestMiddlewareProtocol:
    def test_stub_satisfies_protocol(self):
        mw: Middleware = StubMiddleware()

        assert isinstance(mw, StubMiddleware)
        assert hasattr(mw, "tools")
        assert callable(mw.prompt)

    def test_plain_class_with_tools_and_prompt_satisfies_protocol(self):
        class Custom:
            @property
            def tools(self):
                return []

            def prompt(self, state, config):
                return None

        mw: Middleware = Custom()

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)


class TestAgentKitTools:
    def test_collects_tools_from_all_middleware(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        mw1 = StubMiddleware(tools=[tool_a])
        mw2 = StubMiddleware(tools=[tool_b])

        kit = AgentKit([mw1, mw2])

        assert len(kit.tools) == 2
        assert kit.tools[0].name == "tool_a"
        assert kit.tools[1].name == "tool_b"

    def test_deduplicates_by_name_first_wins(self):
        tool_first = _make_tool("shared", description="first")
        tool_second = _make_tool("shared", description="second")
        mw1 = StubMiddleware(tools=[tool_first])
        mw2 = StubMiddleware(tools=[tool_second])

        kit = AgentKit([mw1, mw2])

        assert len(kit.tools) == 1
        assert kit.tools[0].description == "first"

    def test_tools_cached_same_identity(self):
        kit = AgentKit([StubMiddleware(tools=[_make_tool("t")])])

        first = kit.tools
        second = kit.tools

        assert first is second

    def test_empty_middleware_list_returns_empty_tools(self):
        kit = AgentKit([])

        assert kit.tools == []

    def test_middleware_with_no_tools_returns_empty(self):
        kit = AgentKit([StubMiddleware()])

        assert kit.tools == []


class TestAgentKitPrompt:
    def test_composes_sections_with_double_newline(self):
        mw1 = StubMiddleware(prompt_text="Section A")
        mw2 = StubMiddleware(prompt_text="Section B")

        kit = AgentKit([mw1, mw2])
        result = kit.prompt({}, {})

        assert result == "Section A\n\nSection B"

    def test_skips_middleware_returning_none(self):
        mw1 = StubMiddleware(prompt_text="Only section")
        mw2 = StubMiddleware(prompt_text=None)

        kit = AgentKit([mw1, mw2])
        result = kit.prompt({}, {})

        assert result == "Only section"

    def test_empty_middleware_returns_empty_string(self):
        kit = AgentKit([])
        result = kit.prompt({}, {})

        assert result == ""

    def test_template_prepended_before_middleware_sections(self):
        mw = StubMiddleware(prompt_text="middleware section")

        kit = AgentKit([mw], prompt="System template")
        result = kit.prompt({}, {})

        assert result == "System template\n\nmiddleware section"

    def test_template_from_inline_string(self):
        kit = AgentKit([], prompt="Inline prompt")
        result = kit.prompt({}, {})

        assert result == "Inline prompt"

    def test_template_from_file_path(self):
        fixture_path = FIXTURES / "prompts" / "nodes" / "researcher.md"

        kit = AgentKit([], prompt=fixture_path)
        result = kit.prompt({}, {})

        assert "Research Assistant" in result

    def test_template_list_multiple_concatenated(self):
        researcher = FIXTURES / "prompts" / "nodes" / "researcher.md"
        analyst = FIXTURES / "prompts" / "nodes" / "analyst.md"

        kit = AgentKit([], prompt=[researcher, analyst])
        result = kit.prompt({}, {})

        assert "Research Assistant" in result
        assert "data analyst" in result
        assert "\n\n" in result

    def test_template_list_with_inline_strings(self):
        kit = AgentKit([], prompt=["Part one", "Part two"])
        result = kit.prompt({}, {})

        assert result == "Part one\n\nPart two"


class TestAgentKitIntegration:
    def test_multiple_middleware_composes_tools_and_prompt(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        mw1 = StubMiddleware(tools=[tool_a], prompt_text="Use tool_a for X")
        mw2 = StubMiddleware(tools=[tool_b], prompt_text="Use tool_b for Y")

        kit = AgentKit([mw1, mw2], prompt="You are an agent.")

        assert len(kit.tools) == 2
        assert kit.tools[0].name == "tool_a"
        assert kit.tools[1].name == "tool_b"

        prompt = kit.prompt({}, {})
        assert prompt == "You are an agent.\n\nUse tool_a for X\n\nUse tool_b for Y"
