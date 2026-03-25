"""Tests for WebSearchMiddleware."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.middleware.web_search import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    WebSearchMiddleware,
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


class TestWebSearchMiddlewareConstruction:
    def test_accepts_base_tool_instances(self):
        mock_tool = _make_mock_tool("test_search")

        mw = WebSearchMiddleware(providers=[mock_tool])

        assert mw is not None

    def test_accepts_callable_and_auto_wraps(self):
        def my_search(query: str) -> str:
            return f"results for {query}"

        mw = WebSearchMiddleware(providers=[my_search])

        assert mw is not None

    def test_accepts_async_callable(self):
        async def my_async_search(query: str) -> str:
            return f"async results for {query}"

        mw = WebSearchMiddleware(providers=[my_async_search])

        assert mw is not None

    def test_accepts_mixed_providers(self):
        mock_tool = _make_mock_tool("tool_search")

        def my_search(query: str) -> str:
            return "results"

        mw = WebSearchMiddleware(providers=[mock_tool, my_search])

        assert mw is not None

    def test_empty_providers_uses_default(self):
        mw = WebSearchMiddleware(providers=[])

        assert len(mw.tools) == 1
        assert mw.tools[0].name == "WebSearch"

    def test_invalid_provider_type_raises_type_error(self):
        with pytest.raises(TypeError):
            WebSearchMiddleware(providers=[42])  # type: ignore[list-item]

    def test_invalid_string_provider_raises_type_error(self):
        with pytest.raises(TypeError):
            WebSearchMiddleware(providers=["not_a_provider"])  # type: ignore[list-item]


class TestDuckDuckGoSearchProviderStandalone:
    def test_is_base_tool(self):
        tool = DuckDuckGoSearchProvider()
        assert isinstance(tool, BaseTool)

    def test_default_name(self):
        tool = DuckDuckGoSearchProvider()
        assert tool.name == "DuckDuckGoSearch"

    def test_configurable_max_results(self):
        tool = DuckDuckGoSearchProvider(max_results=3)
        assert tool.max_results == 3

    def test_configurable_headers(self):
        tool = DuckDuckGoSearchProvider(headers={"User-Agent": "custom/1.0"})
        assert tool.headers["User-Agent"] == "custom/1.0"

    def test_default_headers_has_user_agent(self):
        tool = DuckDuckGoSearchProvider()
        assert "User-Agent" in tool.headers

    def test_importable_from_package(self):
        from langchain_agentkit import DuckDuckGoSearchProvider as Imported

        assert Imported is DuckDuckGoSearchProvider



class TestQwantSearchProviderStandalone:
    def test_is_base_tool(self):
        tool = QwantSearchProvider()
        assert isinstance(tool, BaseTool)

    def test_default_name(self):
        tool = QwantSearchProvider()
        assert tool.name == "QwantSearch"

    def test_configurable_max_results(self):
        tool = QwantSearchProvider(max_results=3)
        assert tool.max_results == 3

    def test_configurable_locale(self):
        tool = QwantSearchProvider(locale="fr_FR")
        assert tool.locale == "fr_FR"

    def test_configurable_safesearch(self):
        tool = QwantSearchProvider(safesearch=2)
        assert tool.safesearch == 2

    def test_configurable_headers(self):
        tool = QwantSearchProvider(headers={"User-Agent": "custom/1.0"})
        assert tool.headers["User-Agent"] == "custom/1.0"

    def test_importable_from_package(self):
        from langchain_agentkit import QwantSearchProvider as Imported

        assert Imported is QwantSearchProvider


class TestDefaultProvider:
    def test_no_providers_uses_qwant_default(self):
        """When no providers given, uses built-in Qwant search."""
        mw = WebSearchMiddleware()
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "WebSearch"

    def test_none_providers_uses_default(self):
        """When providers=None, uses built-in Qwant search."""
        mw = WebSearchMiddleware(providers=None)
        assert len(mw.tools) == 1

    def test_empty_list_uses_default(self):
        """When providers=[], uses built-in Qwant search."""
        mw = WebSearchMiddleware(providers=[])
        assert len(mw.tools) == 1

    def test_default_provider_prompt_mentions_qwant(self):
        """Default provider's name appears in the prompt."""
        mw = WebSearchMiddleware()
        prompt = mw.prompt({}, MagicMock())
        assert "qwant" in prompt.lower()


class TestWebSearchMiddlewareTools:
    def test_tools_returns_single_web_search_tool(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        tools = mw.tools

        assert len(tools) == 1
        assert tools[0].name == "WebSearch"

    def test_tools_cached_after_first_access(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        first = mw.tools
        second = mw.tools

        assert first is second


class TestWebSearchMiddlewarePrompt:
    def test_prompt_mentions_provider_names(self):
        mock_tool = _make_mock_tool("brave_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        result = mw.prompt({}, _TEST_RUNTIME)

        assert "brave_search" in result

    def test_prompt_returns_string(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)

    def test_prompt_with_custom_template(self):
        mock_tool = _make_mock_tool("test_search")
        custom_template = "Use {provider_names} for all web searches."
        mw = WebSearchMiddleware(providers=[mock_tool], prompt_template=custom_template)

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)
        assert result is not None

    def test_prompt_with_custom_template_path(self, tmp_path: Path):
        template_file = tmp_path / "prompt.txt"
        template_file.write_text("Search using {provider_names} to find answers.")
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool], prompt_template=template_file)

        result = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result, str)


class TestWebSearchToolExecution:
    async def test_web_search_fans_out_to_all_providers(self):
        provider_a = _make_mock_tool("provider_a", return_value="result from a")
        provider_b = _make_mock_tool("provider_b", return_value="result from b")
        mw = WebSearchMiddleware(providers=[provider_a, provider_b])
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
        mw = WebSearchMiddleware(providers=[good_provider, bad_provider])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert "good result" in output
        assert output is not None

    async def test_all_providers_fail_returns_errors(self):
        provider_a = _make_mock_tool("provider_a")
        provider_a.ainvoke = AsyncMock(side_effect=RuntimeError("a failed"))
        provider_b = _make_mock_tool("provider_b")
        provider_b.ainvoke = AsyncMock(side_effect=RuntimeError("b failed"))
        mw = WebSearchMiddleware(providers=[provider_a, provider_b])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert output is not None
        assert isinstance(output, str)

    async def test_results_attributed_per_provider(self):
        provider_a = _make_mock_tool("provider_a", return_value="result from a")
        provider_b = _make_mock_tool("provider_b", return_value="result from b")
        mw = WebSearchMiddleware(providers=[provider_a, provider_b])
        web_search_tool = mw.tools[0]

        output = await web_search_tool.ainvoke({"query": "test query"})

        assert "## provider_a" in output
        assert "## provider_b" in output

    def test_sync_run_works(self):
        mock_tool = _make_mock_tool("test_search", return_value="sync result")
        mw = WebSearchMiddleware(providers=[mock_tool])
        web_search_tool = mw.tools[0]

        output = web_search_tool.invoke({"query": "test query"})

        assert output is not None
        assert isinstance(output, str)


class TestMiddlewareProtocol:
    def test_implements_middleware_protocol(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        assert hasattr(mw, "tools")
        assert isinstance(mw.tools, list)
        assert callable(mw.prompt)

    def test_no_wrap_tool_call(self):
        mock_tool = _make_mock_tool("test_search")
        mw = WebSearchMiddleware(providers=[mock_tool])

        assert not hasattr(mw, "wrap_tool_call")
