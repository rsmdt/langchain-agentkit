"""Tests for WebFetch — SSRF guard, conversion, bounding, and folding into WebSearchExtension.

The SSRF guard is the security surface, so it is tested exhaustively against a
mocked resolver (no network). Transport is kept thin; the HTML-conversion path
is exercised behind the optional `web` extra.
"""

from unittest.mock import patch

import pytest

from langchain_agentkit.extensions.web_search import WebSearchExtension
from langchain_agentkit.extensions.web_search.tools import web_fetch
from langchain_agentkit.extensions.web_search.tools.web_fetch import (
    BlockedURLError,
    FetchError,
    _bound_tokens,
    _canonical_ip,
    _fetch,
    _is_public,
    _resolve_and_validate,
    _to_markdown,
)


def _addrinfo(*ips: str):
    """Fake socket.getaddrinfo return: one tuple per IP, address in slot [4][0]."""
    return [(None, None, None, "", (ip, 0)) for ip in ips]


class TestCanonicalIP:
    def test_plain_v4(self):
        assert str(_canonical_ip("8.8.8.8")) == "8.8.8.8"

    def test_ipv4_mapped_ipv6_is_unwrapped(self):
        # ::ffff:169.254.169.254 embeds the cloud-metadata v4 address in v6.
        assert str(_canonical_ip("::ffff:169.254.169.254")) == "169.254.169.254"

    def test_6to4_is_unwrapped(self):
        # 2002:a9fe:a9fe:: embeds 169.254.169.254 via 6to4.
        assert str(_canonical_ip("2002:a9fe:a9fe::")) == "169.254.169.254"


class TestIsPublic:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.1",  # private
            "192.168.1.1",  # private
            "172.16.0.1",  # private
            "169.254.169.254",  # link-local (cloud metadata)
            "0.0.0.0",  # unspecified
            "::1",  # v6 loopback
            "fe80::1",  # v6 link-local
            "::ffff:127.0.0.1",  # v4-mapped loopback
            "::ffff:169.254.169.254",  # v4-mapped metadata
            "2002:a9fe:a9fe::",  # 6to4 metadata
        ],
    )
    def test_non_public_rejected(self, ip):
        assert _is_public(ip) is False

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_public_allowed(self, ip):
        assert _is_public(ip) is True


class TestResolveAndValidate:
    def test_public_host_returns_ip(self):
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            assert _resolve_and_validate("example.com") == "93.184.216.34"

    def test_private_resolution_blocked(self):
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("10.0.0.5")),
            pytest.raises(BlockedURLError),
        ):
            _resolve_and_validate("internal.corp")

    def test_metadata_endpoint_blocked(self):
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")),
            pytest.raises(BlockedURLError),
        ):
            _resolve_and_validate("metadata.evil")

    def test_any_private_in_mixed_result_blocks(self):
        # A name that resolves public + private is refused, not gambled on.
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34", "10.0.0.5")),
            pytest.raises(BlockedURLError),
        ):
            _resolve_and_validate("rebind.evil")

    def test_unresolvable_host_blocked(self):
        import socket

        with (
            patch("socket.getaddrinfo", side_effect=socket.gaierror("nope")),
            pytest.raises(BlockedURLError),
        ):
            _resolve_and_validate("does-not-exist.invalid")


class TestFetchGuardRejectsBadSchemes:
    async def test_non_http_scheme_blocked(self):
        tool = WebSearchExtension(fetch=True).tools[1]
        with pytest.raises(BlockedURLError):
            await tool.ainvoke(
                {
                    "name": "WebFetch",
                    "args": {"url": "file:///etc/passwd"},
                    "id": "1",
                    "type": "tool_call",
                }
            )

    async def test_private_target_blocked_end_to_end(self):
        tool = WebSearchExtension(fetch=True).tools[1]
        with (
            patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")),
            pytest.raises(BlockedURLError),
        ):
            await tool.ainvoke(
                {
                    "name": "WebFetch",
                    "args": {"url": "http://localhost/admin"},
                    "id": "1",
                    "type": "tool_call",
                }
            )


class _FakeResponse:
    """Stand-in for http.client.HTTPResponse in _fetch redirect-loop tests."""

    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None):
        self.status = status
        self._body = body
        self._headers = headers or {}
        self._pos = 0

    def getheader(self, name: str, default: str = "") -> str:
        return self._headers.get(name, default)

    def read(self, amt: int) -> bytes:
        chunk = self._body[self._pos : self._pos + amt]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class TestFetchTransport:
    def test_successful_fetch_returns_body(self):
        resp = _FakeResponse(200, b"hello", {"Content-Type": "text/plain"})
        with patch.object(web_fetch, "_request", return_value=resp) as req:
            final_url, content_type, body = _fetch(
                "https://example.com/page", timeout=5, max_bytes=1000
            )
        assert body == b"hello"
        assert content_type == "text/plain"
        assert final_url == "https://example.com/page"
        req.assert_called_once()

    def test_redirect_is_followed_and_revalidated(self):
        # First call redirects; second returns the body. Each _request re-runs the guard.
        first = _FakeResponse(301, headers={"Location": "https://example.com/final"})
        second = _FakeResponse(200, b"arrived", {"Content-Type": "text/plain"})
        with patch.object(web_fetch, "_request", side_effect=[first, second]) as req:
            final_url, _, body = _fetch("https://example.com/start", timeout=5, max_bytes=1000)
        assert body == b"arrived"
        assert final_url == "https://example.com/final"
        assert req.call_count == 2

    def test_redirect_without_location_errors(self):
        resp = _FakeResponse(302, headers={})
        with (
            patch.object(web_fetch, "_request", return_value=resp),
            pytest.raises(FetchError, match="without Location"),
        ):
            _fetch("https://example.com", timeout=5, max_bytes=1000)

    def test_redirect_cap_enforced(self):
        loop = _FakeResponse(301, headers={"Location": "https://example.com/again"})
        with (
            patch.object(web_fetch, "_request", return_value=loop),
            pytest.raises(FetchError, match="too many redirects"),
        ):
            _fetch("https://example.com/again", timeout=5, max_bytes=1000)

    def test_http_error_status_raises(self):
        resp = _FakeResponse(404, b"nope", {"Content-Type": "text/plain"})
        with (
            patch.object(web_fetch, "_request", return_value=resp),
            pytest.raises(FetchError, match="HTTP 404"),
        ):
            _fetch("https://example.com/missing", timeout=5, max_bytes=1000)

    def test_body_read_is_bounded(self):
        resp = _FakeResponse(200, b"x" * 5000, {"Content-Type": "text/plain"})
        with patch.object(web_fetch, "_request", return_value=resp):
            _, _, body = _fetch("https://example.com/big", timeout=5, max_bytes=100)
        assert len(body) == 100


class TestBoundTokens:
    def test_under_budget_untouched(self):
        text = "short"
        assert _bound_tokens(text, 100) == (text, False)

    def test_over_budget_truncated(self):
        text = "x" * 1000
        out, truncated = _bound_tokens(text, 10)  # 10 tokens * 4 chars = 40
        assert truncated is True
        assert len(out) == 40


class TestToMarkdown:
    def test_plain_text_passthrough_no_extra_needed(self):
        # text/* content skips conversion entirely — no markdownify import.
        assert _to_markdown(b"just text", "text/plain") == "just text"

    def test_html_converted_to_markdown(self):
        pytest.importorskip("markdownify")
        out = _to_markdown(
            b"<html><body><h1>Title</h1><p>Body text.</p></body></html>", "text/html"
        )
        assert "Title" in out
        assert "Body text." in out
        assert "<h1>" not in out

    def test_html_without_extra_raises_import_error(self):
        # Simulate the extra being absent: make the markdownify import fail.
        with (
            patch.dict("sys.modules", {"markdownify": None}),
            pytest.raises(ImportError, match="langchain-agentkit\\[web\\]"),
        ):
            _to_markdown(b"<html><h1>x</h1></html>", "text/html")


class TestFoldIntoWebSearchExtension:
    def test_search_only_by_default(self):
        ext = WebSearchExtension()
        assert [t.name for t in ext.tools] == ["WebSearch"]

    def test_fetch_true_adds_webfetch(self):
        ext = WebSearchExtension(fetch=True)
        assert [t.name for t in ext.tools] == ["WebSearch", "WebFetch"]

    def test_fetch_tool_advertises_content_and_artifact(self):
        tool = WebSearchExtension(fetch=True).tools[1]
        assert tool.response_format == "content_and_artifact"

    def test_fetch_true_without_extra_raises_at_construction(self):
        # Fail fast: a kit asking for fetch without the extra is misconfigured now.
        with (
            patch("importlib.util.find_spec", return_value=None),
            pytest.raises(ImportError, match="langchain-agentkit\\[web\\]"),
        ):
            WebSearchExtension(fetch=True)
