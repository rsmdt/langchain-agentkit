"""Tests for the shared shell-command construction helpers."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.backends.shell import parse_read_window, read_window_command, shell_quote

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_agentkit.backends.results import ReadResult


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


def _read(path: Path, offset: int, limit: int) -> ReadResult:
    output = subprocess.run(
        ["sh", "-c", read_window_command(str(path), offset, limit)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return parse_read_window(output)


class TestReadWindow:
    def test_window_and_total_lines(self, tmp_path: Path):
        target = tmp_path / "f.txt"
        target.write_text("a\nb\nc\nd")
        result = _read(target, 1, 2)
        assert result.content == "b\nc\n"
        assert result.total_lines == 4

    def test_negative_offset_clamps_to_start(self, tmp_path: Path):
        target = tmp_path / "f.txt"
        target.write_text("a\nb\n")
        assert _read(target, -4, 1).content == "a\n"

    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_is_empty_read(self, tmp_path: Path, limit: int):
        target = tmp_path / "f.txt"
        target.write_text("a\nb\n")
        result = _read(target, 0, limit)
        assert result.content == ""
        assert result.total_lines == 2

    def test_unparseable_output_is_io_error(self):
        assert parse_read_window("bash: awk: not found\n").error == "io_error"
