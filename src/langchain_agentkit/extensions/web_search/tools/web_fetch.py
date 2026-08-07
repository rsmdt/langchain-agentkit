"""WebFetch — read one URL, convert to markdown, bounded, SSRF-guarded.

The guard is the point of this tool, not the plumbing. Fetching an
attacker-influenced URL is a server-side request forgery (SSRF) primitive: a
model told to "read https://…" can be steered at cloud metadata endpoints
(``169.254.169.254``), loopback admin panels, or internal hosts. Defenses,
all always-on:

- scheme allowlist (``http``/``https`` only);
- resolve the host and reject if *any* resolved address is private, loopback,
  link-local, reserved, multicast, or unspecified — including IPv4-mapped and
  6to4-embedded IPv6 forms that smuggle a private v4 address inside a v6 one;
- **DNS pinning**: connect to the exact address that was validated, so a name
  that resolves public-then-private (rebinding) cannot slip through the gap
  between check and connect. Pinning is per-connection (the validated IP is
  handed to the socket while TLS SNI and certificate validation stay on the
  hostname), so unlike a process-global ``getaddrinfo`` patch it does not
  serialize concurrent fetches;
- every redirect hop is re-validated through the same checks, capped.

Human-approval gating is orthogonal: compose ``HITLExtension`` with
``interrupt_on=["WebFetch"]``. The SSRF guard runs regardless.

Markdown conversion needs the optional ``web`` extra
(``pip install langchain-agentkit[web]``).
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from typing import Any, Literal, override
from urllib.parse import urljoin, urlsplit

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECTS = 5
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_MAX_CONTENT_TOKENS = 25_000
_CHARS_PER_TOKEN = 4  # rough; the cap is deliberately approximate
_READ_CHUNK = 65_536

_WEB_FETCH_DESCRIPTION = """Fetch one URL and return its main content as markdown. Use to read a page you already have the URL for (e.g. a result from WebSearch). Content is truncated to a token budget."""


class FetchError(Exception):
    """A fetch failed for a non-security reason (network, HTTP status, size)."""


class BlockedURLError(FetchError):
    """A URL was refused by the SSRF guard before any request was sent."""


def _canonical_ip(ip_str: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Normalize an address, unwrapping IPv4-mapped and 6to4 IPv6 forms.

    ``::ffff:169.254.169.254`` and ``2002:a9fe:a9fe::`` both embed a private
    v4 address inside a v6 one; without unwrapping, a v6-level ``is_private``
    check passes them through.
    """
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip.sixtofour is not None:
            return ip.sixtofour
    return ip


def _is_public(ip_str: str) -> bool:
    ip = _canonical_ip(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_and_validate(host: str) -> str:
    """Resolve ``host`` and return one validated public IP.

    Rejects if the host resolves to no address, or if *any* resolved address
    is non-public — a name returning a mix of public and private results is
    refused rather than gambled on.
    """
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise BlockedURLError(f"cannot resolve host {host!r}: {e}") from e
    ips = [str(info[4][0]) for info in infos]
    if not ips:
        raise BlockedURLError(f"host {host!r} resolved to no address")
    for ip in ips:
        if not _is_public(ip):
            raise BlockedURLError(f"host {host!r} resolves to non-public address {ip}")
    return ips[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, pinned_ip: str, port: int, *, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    @override
    def connect(self) -> None:
        # Connect to the validated IP; the Host header still carries self.host.
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self, host: str, pinned_ip: str, port: int, *, timeout: float, context: ssl.SSLContext
    ) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self._pinned_ip = pinned_ip
        self._ssl_context = context

    @override
    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        # TLS SNI and certificate validation stay bound to the hostname, not
        # the pinned IP — the connection is to the IP we validated, verified
        # as the host we intended.
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


def _request(url: str, timeout: float, headers: dict[str, str]) -> http.client.HTTPResponse:
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise BlockedURLError(f"scheme {parts.scheme!r} not allowed")
    host = parts.hostname
    if not host:
        raise BlockedURLError(f"no host in URL {url!r}")
    pinned_ip = _resolve_and_validate(host)

    conn: http.client.HTTPConnection
    if parts.scheme == "https":
        conn = _PinnedHTTPSConnection(
            host,
            pinned_ip,
            parts.port or 443,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    else:
        conn = _PinnedHTTPConnection(host, pinned_ip, parts.port or 80, timeout=timeout)

    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    conn.request("GET", path, headers=headers)
    return conn.getresponse()


def _fetch(url: str, *, timeout: float, max_bytes: int) -> tuple[str, str, bytes]:
    """Fetch ``url`` following validated redirects. Returns (final_url, content_type, body)."""
    headers = {
        "User-Agent": "langchain-agentkit-webfetch/1.0",
        "Accept": "text/html,text/markdown,text/plain;q=0.9,*/*;q=0.5",
    }
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        response = _request(current, timeout, headers)
        try:
            if response.status in _REDIRECT_CODES:
                location = response.getheader("Location")
                if not location:
                    raise FetchError(f"redirect {response.status} without Location")
                current = urljoin(current, location)  # re-validated by next _request
                continue
            if response.status >= 400:
                raise FetchError(f"HTTP {response.status} fetching {current}")
            content_type = response.getheader("Content-Type", "")
            body = _read_bounded(response, max_bytes)
            return current, content_type, body
        finally:
            response.close()
    raise FetchError(f"too many redirects (> {_MAX_REDIRECTS}) fetching {url}")


def _read_bounded(response: http.client.HTTPResponse, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = response.read(min(_READ_CHUNK, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _to_markdown(body: bytes, content_type: str) -> str:
    text = body.decode("utf-8", "replace")
    if "html" not in content_type.lower():
        return text  # already text/markdown/plain
    try:
        from markdownify import markdownify  # noqa: PLC0415 — optional [web] extra
    except ImportError as e:
        raise ImportError(
            "WebFetch needs the 'web' extra for HTML conversion: "
            "pip install langchain-agentkit[web]"
        ) from e
    try:
        from readabilipy import simple_json_from_html_string  # noqa: PLC0415

        article = simple_json_from_html_string(text, use_readability=False)
        if article.get("content"):
            text = article["content"]
    except ImportError:
        pass  # readabilipy is optional even within the extra; convert whole doc
    return markdownify(text).strip()


def _bound_tokens(markdown: str, max_content_tokens: int) -> tuple[str, bool]:
    limit = max_content_tokens * _CHARS_PER_TOKEN
    if len(markdown) <= limit:
        return markdown, False
    return markdown[:limit], True


class _WebFetchInput(BaseModel):
    url: str = Field(description="The URL to fetch.")
    max_content_tokens: int = Field(
        default=_DEFAULT_MAX_CONTENT_TOKENS,
        ge=100,
        le=200_000,
        description="Approximate cap on returned content, in tokens. Content beyond it is truncated.",
    )


class _WebFetchTool(BaseTool):
    """Fetch a URL, convert to markdown, bound the result. SSRF-guarded."""

    name: str = "WebFetch"
    description: str = _WEB_FETCH_DESCRIPTION
    args_schema: type[BaseModel] = _WebFetchInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    timeout: float = 30.0

    def _fetch_to_markdown(
        self, url: str, max_content_tokens: int
    ) -> tuple[str, list[dict[str, Any]]]:
        max_bytes = max_content_tokens * _CHARS_PER_TOKEN * 4  # generous pre-convert read cap
        final_url, content_type, body = _fetch(url, timeout=self.timeout, max_bytes=max_bytes)
        markdown = _to_markdown(body, content_type)
        content, truncated = _bound_tokens(markdown, max_content_tokens)
        if truncated:
            content += "\n\n[content truncated at the token budget]"
        artifact = [
            {
                "url": final_url,
                "content_type": content_type,
                "content_length": len(content),
                "truncated": truncated,
            }
        ]
        return content, artifact

    @override
    def _run(
        self, url: str, max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS
    ) -> tuple[str, list[dict[str, Any]]]:
        return self._fetch_to_markdown(url, max_content_tokens)

    @override
    async def _arun(
        self, url: str, max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS
    ) -> tuple[str, list[dict[str, Any]]]:
        import asyncio  # noqa: PLC0415

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_to_markdown, url, max_content_tokens)
