"""Tests for the shared shell-command construction helpers."""

from __future__ import annotations

from langchain_agentkit.backends.shell import shell_quote


class TestShellQuote:
    def test_plain_string(self):
        assert shell_quote("hello") == "'hello'"

    def test_string_with_spaces(self):
        assert shell_quote("hello world") == "'hello world'"

    def test_string_with_single_quote(self):
        # POSIX 'safe' single-quote: 'it'"'"'s'
        assert shell_quote("it's") == "'it'\"'\"'s'"

    def test_string_with_special_chars(self):
        result = shell_quote("$HOME `cmd` \\n")
        # Inside single quotes, $, `, \ are literal
        assert result.startswith("'")
        assert result.endswith("'")
        assert "$HOME" in result

    def test_empty_string(self):
        assert shell_quote("") == "''"

    def test_string_with_newline(self):
        result = shell_quote("line1\nline2")
        assert "\n" in result
