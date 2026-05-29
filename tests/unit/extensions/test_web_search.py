"""Tests for WebSearchExtension."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.extensions.web_search import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebSearchExtension,
)

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


def _make_mock_tool(name: str, return_value: str = "search results") -> MagicMock:
    mock = MagicMock(spec=BaseTool)
    mock.name = name
    mock.ainvoke = AsyncMock(return_value=return_value)
    mock.invoke = MagicMock(return_value=return_value)
    return mock


class TestWebSearchExtensionConstruction:
    def test_empty_providers_uses_default(self):
        mw = WebSearchExtension(providers=[])

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "WebSearch"

    def test_invalid_provider_type_raises_type_error(self):
        with pytest.raises(TypeError):
            WebSearchExtension(providers=[42])  # type: ignore[list-item]

    def test_invalid_string_provider_raises_type_error(self):
        with pytest.raises(TypeError):
            WebSearchExtension(providers=["not_a_provider"])  # type: ignore[list-item]


class TestDuckDuckGoSearchProviderStandalone:
    def test_default_name(self):
        tool = DuckDuckGoSearchProvider()
        assert tool.name == "DuckDuckGoSearch"

    def test_default_headers_has_user_agent(self):
        tool = DuckDuckGoSearchProvider()
        assert "User-Agent" in tool.headers


class TestQwantSearchProviderStandalone:
    def test_default_name(self):
        tool = QwantSearchProvider()
        assert tool.name == "QwantSearch"


class TestDefaultProvider:
    def test_no_providers_uses_qwant_default(self):
        """When no providers given, uses built-in Qwant search."""
        mw = WebSearchExtension()
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "WebSearch"

    def test_none_providers_uses_default(self):
        """When providers=None, uses built-in Qwant search."""
        mw = WebSearchExtension(providers=None)
        assert len(mw.tools) == 1

    def test_empty_list_uses_default(self):
        """When providers=[], uses built-in Qwant search."""
        mw = WebSearchExtension(providers=[])
        assert len(mw.tools) == 1

    def test_default_provider_prompt_emits_sources_norm(self):
        """WebSearchExtension contributes the Sources behavioral norm."""
        mw = WebSearchExtension()
        prompt = mw.prompt({}, MagicMock())
        assert prompt is not None
        assert "Sources" in prompt


class TestWebSearchExtensionTools:
    def test_tools_returns_single_web_search_tool(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchExtension(providers=[mock_tool])

        tools = mw.tools

        assert len(tools) == 1
        assert tools[0].name == "WebSearch"

    def test_tools_cached_after_first_access(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchExtension(providers=[mock_tool])

        first = mw.tools
        second = mw.tools

        assert first is second


class TestWebSearchExtensionPrompt:
    def test_prompt_emits_sources_norm(self):
        """The Sources behavioral norm is contributed regardless of provider."""
        mock_tool = _make_mock_tool("brave_search")
        mw = WebSearchExtension(providers=[mock_tool])

        prompt = mw.prompt({}, _TEST_RUNTIME)
        assert prompt is not None
        assert "Sources" in prompt

    def test_prompt_emits_sources_norm_with_custom_template(self):
        """Custom template no longer affects the (norm-only) prompt output."""
        mock_tool = _make_mock_tool("test_search")
        custom_template = "Use {provider_names} for all web searches."
        mw = WebSearchExtension(providers=[mock_tool], prompt_template=custom_template)

        prompt = mw.prompt({}, _TEST_RUNTIME)
        assert prompt is not None
        assert "Sources" in prompt


class TestWebSearchToolExecution:
    async def test_web_search_fans_out_to_all_providers(self):
        provider_a = _make_mock_tool("provider_a", return_value="result from a")
        provider_b = _make_mock_tool("provider_b", return_value="result from b")
        mw = WebSearchExtension(providers=[provider_a, provider_b])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        provider_a.ainvoke.assert_called_once()
        provider_b.ainvoke.assert_called_once()
        assert "result from a" in output
        assert "result from b" in output

    async def test_provider_error_captured_not_raised(self):
        good_provider = _make_mock_tool("good_provider", return_value="good result")
        bad_provider = _make_mock_tool("bad_provider")
        bad_provider.ainvoke = AsyncMock(side_effect=RuntimeError("provider failed"))
        mw = WebSearchExtension(providers=[good_provider, bad_provider])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert "good result" in output
        assert output is not None

    async def test_all_providers_fail_returns_errors(self):
        provider_a = _make_mock_tool("provider_a")
        provider_a.ainvoke = AsyncMock(side_effect=RuntimeError("a failed"))
        provider_b = _make_mock_tool("provider_b")
        provider_b.ainvoke = AsyncMock(side_effect=RuntimeError("b failed"))
        mw = WebSearchExtension(providers=[provider_a, provider_b])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert output is not None
        assert isinstance(output, str)

    async def test_results_attributed_per_provider(self):
        provider_a = _make_mock_tool("provider_a", return_value="result from a")
        provider_b = _make_mock_tool("provider_b", return_value="result from b")
        mw = WebSearchExtension(providers=[provider_a, provider_b])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert "## provider_a" in output
        assert "## provider_b" in output

    def test_sync_run_works(self):
        mock_tool = _make_mock_tool("test_search", return_value="sync result")
        mw = WebSearchExtension(providers=[mock_tool])
        web_search_tool = mw.tools[0]

        output = web_search_tool.invoke({"query": "test query"})

        assert output is not None
        assert isinstance(output, str)


class TestExtensionProtocol:
    def test_implements_extension_protocol(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchExtension(providers=[mock_tool])

        assert hasattr(mw, "tools")
        assert isinstance(mw.tools, list)
        assert callable(mw.prompt)

    def test_no_wrap_tool_call(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchExtension(providers=[mock_tool])

        assert not hasattr(mw, "wrap_tool_call")
