"""Tests for WebExtension."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolCall
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.extensions.web import (
    DuckDuckGoSearchProvider,
    QwantSearchProvider,
    SearchProvider,
    SearchResult,
    WebExtension,
)

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class _FakeProvider(SearchProvider):
    """In-memory provider returning canned records, or raising."""

    def __init__(self, name: str, results=None, error: Exception | None = None) -> None:
        self.name = name
        self._results = results or []
        self._error = error
        self.calls = 0

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._results[:max_results])


def _result(url: str, title: str = "t", snippet: str = "s") -> SearchResult:
    return SearchResult(url=url, title=title, snippet=snippet)


def _tool_call(query: str = "q", **args) -> ToolCall:
    """A proper ToolCall payload — makes a content_and_artifact tool return a ToolMessage."""
    return ToolCall(name="WebSearch", args={"query": query, **args}, id="call-1", type="tool_call")


class TestSearchResult:
    def test_as_dict_is_serializable(self):
        r = SearchResult(url="https://x", title="T", snippet="S", provider="P")
        assert r.as_dict() == {
            "url": "https://x",
            "title": "T",
            "snippet": "S",
            "published": None,
            "score": None,
            "provider": "P",
        }


class TestWebExtensionConstruction:
    def test_empty_providers_exposes_fetch_only(self):
        assert [t.name for t in WebExtension(providers=[]).tools] == ["WebFetch"]

    def test_none_providers_exposes_fetch_only(self):
        assert [t.name for t in WebExtension(providers=None).tools] == ["WebFetch"]

    def test_providers_add_websearch_before_webfetch(self):
        ext = WebExtension(providers=[DuckDuckGoSearchProvider()])
        assert [t.name for t in ext.tools] == ["WebSearch", "WebFetch"]
        assert ext.tools[0].providers[0].name == "DuckDuckGoSearch"

    def test_invalid_provider_type_raises_type_error(self):
        with pytest.raises(TypeError):
            WebExtension(providers=[42])  # type: ignore[list-item]

    def test_basetool_provider_rejected(self):
        """Direct BaseTool providers are no longer accepted — providers are SearchProvider."""
        with pytest.raises(TypeError):
            WebExtension(providers=[MagicMock()])  # type: ignore[list-item]


class TestDuckDuckGoSearchProviderStandalone:
    def test_default_name(self):
        assert DuckDuckGoSearchProvider().name == "DuckDuckGoSearch"

    def test_instant_mode_is_default(self):
        assert DuckDuckGoSearchProvider().use_html is False

    def test_instant_mode_uses_platform_user_agent(self):
        assert "Mozilla" not in DuckDuckGoSearchProvider().headers["User-Agent"]

    def test_html_mode_defaults_to_browser_user_agent(self):
        assert "Mozilla" in DuckDuckGoSearchProvider(use_html=True).headers["User-Agent"]

    def test_explicit_headers_override_html_default(self):
        tool = DuckDuckGoSearchProvider(use_html=True, headers={"User-Agent": "custom"})
        assert tool.headers["User-Agent"] == "custom"

    def test_html_mode_parses_serp_results_as_records(self):
        sample_html = """
        <a class="result__a" href="https://example.com/a">First Result</a>
        <a class="result__snippet" href="https://example.com/a">Snippet one here.</a>
        <a class="result__a" href="https://example.com/b">Second Result</a>
        <a class="result__snippet" href="https://example.com/b">Snippet two.</a>
        """

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return sample_html.encode()

        tool = DuckDuckGoSearchProvider(use_html=True)
        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            results = tool.search("any query")

        assert [r.url for r in results] == ["https://example.com/a", "https://example.com/b"]
        assert results[0].title == "First Result"
        assert results[0].snippet == "Snippet one here."
        assert results[0].published is None

    def test_html_mode_no_results_returns_empty(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"<html><body>nothing here</body></html>"

        tool = DuckDuckGoSearchProvider(use_html=True)
        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            assert tool.search("any query") == []

    def test_max_results_is_honored(self):
        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return (
                    b'{"Abstract":"","RelatedTopics":['
                    b'{"Text":"one","FirstURL":"https://x/1"},'
                    b'{"Text":"two","FirstURL":"https://x/2"},'
                    b'{"Text":"three","FirstURL":"https://x/3"}]}'
                )

        tool = DuckDuckGoSearchProvider()
        with patch("urllib.request.urlopen", return_value=_FakeResponse()):
            assert len(tool.search("q", max_results=2)) == 2


class TestQwantSearchProviderStandalone:
    def test_default_name(self):
        assert QwantSearchProvider().name == "QwantSearch"


class TestDefaultProviderPrompt:
    def test_prompt_emits_sources_norm_when_tool_present(self):
        mw = WebExtension(providers=[_FakeProvider("x")])
        prompt = mw.prompt({}, _TEST_RUNTIME, tools=frozenset({"WebSearch"}))
        assert prompt is not None
        assert "Sources" in prompt

    def test_prompt_emits_sources_norm_outside_composed_kit(self):
        """Empty tools set means standalone call — keep the norm."""
        assert "Sources" in WebExtension(providers=[_FakeProvider("x")]).prompt({}, _TEST_RUNTIME)

    def test_prompt_suppressed_when_tool_shadowed(self):
        """When WebSearch is composed away (shadowed), the norm is not contributed."""
        mw = WebExtension(providers=[_FakeProvider("x")])
        assert mw.prompt({}, _TEST_RUNTIME, tools=frozenset({"OtherTool"})) is None

    def test_prompt_suppressed_without_providers(self):
        assert WebExtension().prompt({}, _TEST_RUNTIME) is None


class TestFallbackMode:
    async def test_single_provider_returns_records(self):
        p = _FakeProvider("solo", [_result("https://a"), _result("https://b")])
        tool = WebExtension(providers=[p]).tools[0]

        msg = await tool.ainvoke(_tool_call())

        assert "https://a" in msg.content
        assert [r["url"] for r in msg.artifact] == ["https://a", "https://b"]
        assert msg.artifact[0]["provider"] == "solo"

    async def test_first_nonempty_provider_wins(self):
        empty = _FakeProvider("empty", [])
        full = _FakeProvider("full", [_result("https://a")])
        tool = WebExtension(providers=[empty, full]).tools[0]

        msg = await tool.ainvoke(_tool_call())

        assert empty.calls == 1
        assert full.calls == 1
        assert [r["url"] for r in msg.artifact] == ["https://a"]

    async def test_all_providers_fail_raises(self):
        """Terminal failure propagates so ResilienceExtension can handle it."""
        a = _FakeProvider("a", error=RuntimeError("a failed"))
        b = _FakeProvider("b", error=RuntimeError("b failed"))
        tool = WebExtension(providers=[a, b]).tools[0]

        with pytest.raises(RuntimeError):
            await tool.ainvoke(_tool_call())

    async def test_empty_results_render_no_results_message(self):
        tool = WebExtension(providers=[_FakeProvider("empty", [])]).tools[0]
        msg = await tool.ainvoke(_tool_call())
        assert msg.content == "No results found."
        assert msg.artifact == []


class TestFanOutMode:
    async def test_merges_and_dedupes_by_url(self):
        a = _FakeProvider("a", [_result("https://dup"), _result("https://a-only")])
        b = _FakeProvider("b", [_result("https://dup"), _result("https://b-only")])
        tool = WebExtension(providers=[a, b], fan_out=True).tools[0]

        msg = await tool.ainvoke(_tool_call())

        urls = [r["url"] for r in msg.artifact]
        assert urls.count("https://dup") == 1
        assert set(urls) == {"https://dup", "https://a-only", "https://b-only"}

    async def test_ranks_by_score_descending(self):
        a = _FakeProvider("a", [SearchResult(url="https://low", score=0.1)])
        b = _FakeProvider("b", [SearchResult(url="https://high", score=0.9)])
        tool = WebExtension(providers=[a, b], fan_out=True).tools[0]

        msg = await tool.ainvoke(_tool_call())

        assert [r["url"] for r in msg.artifact] == ["https://high", "https://low"]

    async def test_partial_failure_still_returns(self):
        good = _FakeProvider("good", [_result("https://a")])
        bad = _FakeProvider("bad", error=RuntimeError("boom"))
        tool = WebExtension(providers=[good, bad], fan_out=True).tools[0]

        msg = await tool.ainvoke(_tool_call())

        assert [r["url"] for r in msg.artifact] == ["https://a"]

    async def test_all_fail_raises(self):
        a = _FakeProvider("a", error=RuntimeError("a"))
        b = _FakeProvider("b", error=RuntimeError("b"))
        tool = WebExtension(providers=[a, b], fan_out=True).tools[0]

        with pytest.raises(RuntimeError):
            await tool.ainvoke(_tool_call())

    async def test_max_results_caps_merged_list(self):
        a = _FakeProvider("a", [_result("https://1"), _result("https://2")])
        b = _FakeProvider("b", [_result("https://3"), _result("https://4")])
        tool = WebExtension(providers=[a, b], fan_out=True).tools[0]

        msg = await tool.ainvoke(_tool_call(max_results=3))

        assert len(msg.artifact) == 3


class TestSyncExecution:
    def test_sync_run_returns_records(self):
        p = _FakeProvider("solo", [_result("https://a")])
        tool = WebExtension(providers=[p]).tools[0]
        msg = tool.invoke(_tool_call())
        assert [r["url"] for r in msg.artifact] == ["https://a"]


class TestExtensionProtocol:
    def test_implements_extension_protocol(self):
        mw = WebExtension(providers=[_FakeProvider("x")])
        assert hasattr(mw, "tools")
        assert isinstance(mw.tools, list)
        assert callable(mw.prompt)

    def test_tool_advertises_content_and_artifact(self):
        tool = WebExtension(providers=[_FakeProvider("x")]).tools[0]
        assert tool.response_format == "content_and_artifact"
