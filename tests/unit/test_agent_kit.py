"""Tests for Extension protocol and AgentKit composition engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from langchain_agentkit.agent_kit import AgentKit, run_extension_setup

if TYPE_CHECKING:
    from langchain_agentkit.extensions import Extension

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_tool(name: str, description: str = "stub") -> StructuredTool:
    def _fn(x: str) -> str:
        return x

    return StructuredTool.from_function(func=_fn, name=name, description=description)


class StubExtension:
    def __init__(self, tools=None, prompt_text=None):
        self._tools = tools or []
        self._prompt_text = prompt_text

    @property
    def tools(self):
        return self._tools

    def prompt(self, state, config):
        return self._prompt_text


class TestExtensionProtocol:
    def test_stub_satisfies_protocol(self):
        mw: Extension = StubExtension()

        assert isinstance(mw, StubExtension)
        assert hasattr(mw, "tools")
        assert callable(mw.prompt)

    def test_plain_class_with_tools_and_prompt_satisfies_protocol(self):
        class Custom:
            @property
            def tools(self):
                return []

            def prompt(self, state, config):
                return None

        mw: Extension = Custom()

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)


class TestAgentKitTools:
    def test_collects_tools_from_all_extensions(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        mw1 = StubExtension(tools=[tool_a])
        mw2 = StubExtension(tools=[tool_b])

        kit = AgentKit(extensions=[mw1, mw2])

        assert len(kit.tools) == 2
        assert kit.tools[0].name == "tool_a"
        assert kit.tools[1].name == "tool_b"

    def test_deduplicates_by_name_first_wins(self):
        tool_first = _make_tool("shared", description="first")
        tool_second = _make_tool("shared", description="second")
        mw1 = StubExtension(tools=[tool_first])
        mw2 = StubExtension(tools=[tool_second])

        kit = AgentKit(extensions=[mw1, mw2])

        assert len(kit.tools) == 1
        assert kit.tools[0].description == "first"

    def test_tools_cached_same_identity(self):
        kit = AgentKit(extensions=[StubExtension(tools=[_make_tool("t")])])

        first = kit.tools
        second = kit.tools

        assert first is second

    def test_empty_extensions_list_returns_empty_tools(self):
        kit = AgentKit(extensions=[])

        assert kit.tools == []

    def test_extensions_with_no_tools_returns_empty(self):
        kit = AgentKit(extensions=[StubExtension()])

        assert kit.tools == []

    def test_deduplicates_preserves_order_first_wins(self):
        """First extensions's tool wins on name collision, order preserved."""
        tool_a = _make_tool("a")
        tool_b_first = _make_tool("b", description="first_b")
        tool_c = _make_tool("c")
        tool_b_second = _make_tool("b", description="second_b")

        mw1 = StubExtension(tools=[tool_a, tool_b_first])
        mw2 = StubExtension(tools=[tool_b_second, tool_c])

        kit = AgentKit(extensions=[mw1, mw2])

        assert len(kit.tools) == 3
        names = [t.name for t in kit.tools]
        assert names == ["a", "b", "c"]
        assert kit.tools[1].description == "first_b"


class TestAgentKitPrompt:
    def test_composes_sections_with_double_newline(self):
        mw1 = StubExtension(prompt_text="Section A")
        mw2 = StubExtension(prompt_text="Section B")

        kit = AgentKit(extensions=[mw1, mw2])
        result = kit.prompt({}, {})

        assert result == "Section A\n\nSection B"

    def test_skips_extensions_returning_none(self):
        mw1 = StubExtension(prompt_text="Only section")
        mw2 = StubExtension(prompt_text=None)

        kit = AgentKit(extensions=[mw1, mw2])
        result = kit.prompt({}, {})

        assert result == "Only section"

    def test_empty_extensions_returns_empty_string(self):
        kit = AgentKit(extensions=[])
        result = kit.prompt({}, {})

        assert result == ""

    def test_template_prepended_before_extensions_sections(self):
        mw = StubExtension(prompt_text="extensions section")

        kit = AgentKit(extensions=[mw], prompt="System template")
        result = kit.prompt({}, {})

        assert result == "System template\n\nextensions section"

    def test_template_from_inline_string(self):
        kit = AgentKit(extensions=[], prompt="Inline prompt")
        result = kit.prompt({}, {})

        assert result == "Inline prompt"

    def test_template_from_file_path(self):
        fixture_path = FIXTURES / "prompts" / "nodes" / "researcher.md"

        kit = AgentKit(extensions=[], prompt=fixture_path)
        result = kit.prompt({}, {})

        assert "Research Assistant" in result

    def test_template_list_multiple_concatenated(self):
        researcher = FIXTURES / "prompts" / "nodes" / "researcher.md"
        analyst = FIXTURES / "prompts" / "nodes" / "analyst.md"

        kit = AgentKit(extensions=[], prompt=[researcher, analyst])
        result = kit.prompt({}, {})

        assert "Research Assistant" in result
        assert "data analyst" in result
        assert "\n\n" in result

    def test_template_list_with_inline_strings(self):
        kit = AgentKit(extensions=[], prompt=["Part one", "Part two"])
        result = kit.prompt({}, {})

        assert result == "Part one\n\nPart two"

    def test_nonexistent_path_treated_as_inline_string(self):
        """A string that looks like a path but doesn't exist is treated as inline."""
        kit = AgentKit(extensions=[], prompt="nonexistent/path/prompt.md")
        result = kit.prompt({}, {})

        assert result == "nonexistent/path/prompt.md"


class TestAgentKitIntegration:
    def test_multiple_extensions_composes_tools_and_prompt(self):
        tool_a = _make_tool("tool_a")
        tool_b = _make_tool("tool_b")
        mw1 = StubExtension(tools=[tool_a], prompt_text="Use tool_a for X")
        mw2 = StubExtension(tools=[tool_b], prompt_text="Use tool_b for Y")

        kit = AgentKit(extensions=[mw1, mw2], prompt="You are an agent.")

        assert len(kit.tools) == 2
        assert kit.tools[0].name == "tool_a"
        assert kit.tools[1].name == "tool_b"

        prompt = kit.prompt({}, {})
        assert prompt == "You are an agent.\n\nUse tool_a for X\n\nUse tool_b for Y"


class DictPromptExtension:
    """Stub extension that returns a dict with prompt/reminder from prompt()."""

    def __init__(self, prompt_text="", reminder=""):
        self._prompt_text = prompt_text
        self._reminder = reminder

    @property
    def tools(self):
        return []

    def prompt(self, state, config):
        return {"prompt": self._prompt_text, "reminder": self._reminder}


class NoneExtension:
    """Stub extension that returns None from prompt()."""

    @property
    def tools(self):
        return []

    def prompt(self, state, config):
        return None


class TestDictPromptReturn:
    """Dict return type with prompt/reminder from extensions."""

    def test_str_return_collected_in_prompt(self):
        mw = StubExtension(prompt_text="Hello")
        kit = AgentKit(extensions=[mw])

        result = kit.prompt({}, {})

        assert result == "Hello"

    def test_prompt_contribution_prompt_field_collected(self):
        mw = DictPromptExtension(prompt_text="Guidance")
        kit = AgentKit(extensions=[mw])

        result = kit.prompt({}, {})

        assert "Guidance" in result

    def test_prompt_contribution_empty_prompt_skipped(self):
        mw = DictPromptExtension(prompt_text="", reminder="something")
        kit = AgentKit(extensions=[mw])

        result = kit.prompt({}, {})

        assert result == ""

    def test_prompt_contribution_reminder_not_in_prompt(self):
        mw = DictPromptExtension(prompt_text="", reminder="Ephemeral")
        kit = AgentKit(extensions=[mw])

        result = kit.prompt({}, {})

        assert "Ephemeral" not in result

    def test_system_reminder_collects_reminder_field(self):
        mw = DictPromptExtension(reminder="Status: 3 pending")
        kit = AgentKit(extensions=[mw])

        result = kit.system_reminder({}, {})

        assert "Status: 3 pending" in result

    def test_system_reminder_empty_when_no_reminders(self):
        mw = StubExtension(prompt_text="Just a prompt")
        kit = AgentKit(extensions=[mw])

        result = kit.system_reminder({}, {})

        assert result == ""

    def test_system_reminder_joins_multiple_reminders(self):
        mw1 = DictPromptExtension(reminder="Reminder A")
        mw2 = DictPromptExtension(reminder="Reminder B")
        kit = AgentKit(extensions=[mw1, mw2])

        result = kit.system_reminder({}, {})

        assert result == "Reminder A\n\nReminder B"

    def test_mixed_str_and_prompt_contribution(self):
        mw1 = StubExtension(prompt_text="String section")
        mw2 = DictPromptExtension(prompt_text="Contribution section")
        kit = AgentKit(extensions=[mw1, mw2])

        result = kit.prompt({}, {})

        assert "String section" in result
        assert "Contribution section" in result

    def test_collect_contributions_single_pass(self):
        mw = DictPromptExtension(prompt_text="P", reminder="R")
        kit = AgentKit(extensions=[mw])

        prompt_parts, reminder_parts = kit._collect_contributions({}, {})

        assert "P" in prompt_parts
        assert "R" in reminder_parts

    def test_none_return_handled(self):
        mw1 = NoneExtension()
        mw2 = StubExtension(prompt_text="Valid")
        kit = AgentKit(extensions=[mw1, mw2])

        result = kit.prompt({}, {})

        assert result == "Valid"


class TestInjectSystemReminder:
    def test_empty_reminder_returns_unchanged_state(self):
        from langchain_agentkit._graph_builder import _inject_system_reminder

        state = {"messages": []}

        result = _inject_system_reminder(state, "")

        assert result is state

    def test_non_empty_reminder_appends_human_message(self):
        from langchain_core.messages import HumanMessage

        from langchain_agentkit._graph_builder import _inject_system_reminder

        state = {"messages": []}

        result = _inject_system_reminder(state, "hello")

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)

    def test_reminder_wrapped_in_system_reminder_tags(self):
        from langchain_agentkit._graph_builder import _inject_system_reminder

        result = _inject_system_reminder({"messages": []}, "hello")

        content = result["messages"][0].content
        assert content.startswith("<system-reminder>")
        assert content.endswith("</system-reminder>")
        assert "hello" in content

    def test_original_state_not_mutated(self):
        from langchain_agentkit._graph_builder import _inject_system_reminder

        original_messages = [{"role": "user", "content": "hi"}]
        state = {"messages": original_messages, "other": "data"}

        result = _inject_system_reminder(state, "reminder")

        assert len(original_messages) == 1
        assert state["messages"] is original_messages
        assert result is not state

    def test_existing_messages_preserved(self):
        from langchain_core.messages import HumanMessage

        from langchain_agentkit._graph_builder import _inject_system_reminder

        existing = HumanMessage(content="existing")
        state = {"messages": [existing]}

        result = _inject_system_reminder(state, "reminder")

        assert len(result["messages"]) == 2
        assert result["messages"][0] is existing


class TestSetupLifecycle:
    """Test the setup() lifecycle hook and introspection-based dispatch."""

    async def test_setup_called_with_extensions(self):
        from langchain_agentkit.extension import Extension

        received: list[object] = []

        class Recorder(Extension):
            def setup(self, *, extensions, **_):
                received.extend(extensions)

        ext = Recorder()
        kit = AgentKit(extensions=[ext])
        await run_extension_setup(kit)

        assert ext in received
        assert len(received) == len(kit._extensions)

    async def test_setup_receives_prompt(self):
        from langchain_agentkit.extension import Extension

        captured: dict[str, object] = {}

        class PromptCapture(Extension):
            def setup(self, *, prompt, **_):
                captured["prompt"] = prompt

        kit = AgentKit(extensions=[PromptCapture()], prompt="Base prompt.")
        await run_extension_setup(kit)

        assert captured["prompt"] == "Base prompt."

    async def test_setup_only_receives_declared_kwargs(self):
        """Introspection should pass only what the extension's signature declares."""
        from langchain_agentkit.extension import Extension

        seen: dict[str, object] = {}

        class MinimalSetup(Extension):
            def setup(self, *, extensions):
                seen["extensions"] = extensions
                # prompt not declared — should not be injected

        kit = AgentKit(extensions=[MinimalSetup()], prompt="Hello")
        await run_extension_setup(kit)

        assert "extensions" in seen
        # MinimalSetup's signature only declares `extensions` — no other kwargs
        # are passed to it (introspection filters them out)

    async def test_setup_with_var_keyword_receives_all(self):
        from langchain_agentkit.extension import Extension

        captured: dict[str, object] = {}

        class VarKwargs(Extension):
            def setup(self, **kwargs):
                captured.update(kwargs)

        kit = AgentKit(extensions=[VarKwargs()], prompt="Hello")
        await run_extension_setup(kit)

        assert "extensions" in captured
        assert "prompt" in captured
        assert captured["prompt"] == "Hello"

    async def test_setup_default_noop(self):
        """Extension with no setup() override should not error."""
        from langchain_agentkit.extension import Extension

        class Plain(Extension):
            pass

        # Should not raise
        kit = AgentKit(extensions=[Plain()])
        await run_extension_setup(kit)

    def test_resolve_model_via_extension(self):
        """kit.resolve_model() should find a model_resolver on any extension."""
        from langchain_agentkit.extension import Extension

        class ResolverExt(Extension):
            model_resolver = staticmethod(lambda name: f"resolved:{name}")

        kit = AgentKit(extensions=[ResolverExt()])

        assert kit.resolve_model("gpt-4o") == "resolved:gpt-4o"

    def test_resolve_model_raises_when_no_resolver(self):
        import pytest

        kit = AgentKit(extensions=[])

        with pytest.raises(ValueError, match="no model_resolver is configured"):
            kit.resolve_model("gpt-4o")

    def test_resolve_model_passes_through_non_string(self):
        from unittest.mock import MagicMock

        kit = AgentKit(extensions=[])
        obj = MagicMock()

        assert kit.resolve_model(obj) is obj

    def test_model_resolver_fallback_chain_resolver_first(self):
        """Kit-level model_resolver takes priority over extension resolver."""
        from langchain_agentkit.extension import Extension

        class ResolverExt(Extension):
            model_resolver = staticmethod(lambda name: f"ext:{name}")

        kit = AgentKit(
            extensions=[ResolverExt()],
            model="gpt-4o",
            model_resolver=lambda name: f"kit:{name}",
        )

        assert kit.model == "kit:gpt-4o"

    def test_model_resolver_fallback_to_extension(self):
        """Falls back to extension resolver when no kit-level resolver."""
        from langchain_agentkit.extension import Extension

        class ResolverExt(Extension):
            model_resolver = staticmethod(lambda name: f"ext:{name}")

        kit = AgentKit(extensions=[ResolverExt()], model="gpt-4o")

        assert kit.model == "ext:gpt-4o"

    def test_model_property_non_string(self):
        """model property returns the raw model when it's not a string."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        kit = AgentKit(extensions=[], model=mock_llm)

        assert kit.model is mock_llm

    def test_model_property_none_when_not_set(self):
        """model property returns None when no model configured."""
        kit = AgentKit(extensions=[])

        assert kit.model is None


class TestAgentKitCompile:
    """Tests for kit.compile(handler)."""

    def test_compile_returns_state_graph(self):
        from unittest.mock import MagicMock

        from langgraph.graph import StateGraph

        mock_llm = MagicMock()
        kit = AgentKit(extensions=[], model=mock_llm)

        async def handler(state, *, llm):
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content="ok")]}

        graph = kit.compile(handler)

        assert isinstance(graph, StateGraph)

    def test_compile_merges_user_and_extension_tools(self):
        from unittest.mock import MagicMock

        from langchain_core.tools import StructuredTool

        user_tool = StructuredTool.from_function(
            func=lambda x: x, name="user_tool", description="user"
        )
        ext_tool = StructuredTool.from_function(
            func=lambda x: x, name="ext_tool", description="ext"
        )

        class ToolExt:
            @property
            def tools(self):
                return [ext_tool]

            def prompt(self, state, runtime):
                return None

        mock_llm = MagicMock()
        kit = AgentKit(extensions=[ToolExt()], tools=[user_tool], model=mock_llm)

        # Both tools should be in kit.tools
        tool_names = [t.name for t in kit.tools]
        assert "user_tool" in tool_names
        assert "ext_tool" in tool_names

    def test_hooks_property_returns_hook_runner(self):
        from langchain_agentkit.hook_runner import HookRunner

        kit = AgentKit(extensions=[])

        assert isinstance(kit.hooks, HookRunner)
