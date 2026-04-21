"""Tests for the PromptTemplateExtension (I4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.extensions.prompt_templates import (
    PromptTemplate,
    PromptTemplateError,
    PromptTemplateExtension,
    expand_template,
    parse_args,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Frontmatter -> PromptTemplate
# ---------------------------------------------------------------------------


class TestFromFrontmatter:
    def test_minimal_valid(self) -> None:
        t = PromptTemplate.from_frontmatter(
            {"name": "review", "description": "Review a file."}, "body"
        )
        assert t.name == "review"
        assert t.description == "Review a file."
        assert t.argument_hint == ""
        assert t.body == "body"

    def test_argument_hint_variants(self) -> None:
        for key in ("argument-hint", "argumentHint", "argument_hint"):
            t = PromptTemplate.from_frontmatter(
                {"name": "r", "description": "d", key: "<x>"}, "body"
            )
            assert t.argument_hint == "<x>"

    def test_extra_metadata_preserved(self) -> None:
        t = PromptTemplate.from_frontmatter(
            {"name": "r", "description": "d", "author": "rsmdt"}, "body"
        )
        assert t.metadata == {"author": "rsmdt"}

    @pytest.mark.parametrize(
        "bad_name", ["", "UPPER", "has space", "leading-", "trailing-", "a" * 65]
    )
    def test_invalid_names_rejected(self, bad_name: str) -> None:
        with pytest.raises(PromptTemplateError):
            PromptTemplate.from_frontmatter({"name": bad_name, "description": "d"}, "body")

    def test_missing_description_rejected(self) -> None:
        with pytest.raises(PromptTemplateError):
            PromptTemplate.from_frontmatter({"name": "ok"}, "body")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_empty(self) -> None:
        parsed = parse_args("")
        assert parsed.positional == ()
        assert parsed.count == 0

    def test_whitespace_splits(self) -> None:
        assert parse_args("one two three").positional == ("one", "two", "three")

    def test_double_quoted_preserves_spaces(self) -> None:
        assert parse_args('one "two words" three').positional == (
            "one",
            "two words",
            "three",
        )

    def test_single_quoted_preserves_spaces(self) -> None:
        assert parse_args("one 'two words' three").positional == (
            "one",
            "two words",
            "three",
        )

    def test_escaped_double_quote(self) -> None:
        assert parse_args(r'a "b\"c" d').positional == ("a", 'b"c', "d")

    def test_sequence_input_passthrough(self) -> None:
        parsed = parse_args(["a", "b with space"])
        assert parsed.positional == ("a", "b with space")

    def test_unterminated_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_args('unterminated "quote')


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------


class TestExpandTemplate:
    def test_simple_positional(self) -> None:
        assert expand_template("Hello $1!", ["world"]) == "Hello world!"

    def test_braced_positional(self) -> None:
        args = [str(i) for i in range(12)]
        assert expand_template("${11}", args) == "10"

    def test_at_all(self) -> None:
        assert expand_template("all: $@", ["a", "b", "c"]) == "all: a b c"

    def test_at_slice(self) -> None:
        args = ["a", "b", "c", "d", "e"]
        assert expand_template("${@:2:2}", args) == "b c"

    def test_slice_out_of_range_empty(self) -> None:
        assert expand_template("${@:10:3}", ["a", "b"]) == ""

    def test_missing_arg_expands_empty(self) -> None:
        assert expand_template("$1-$2-$3", ["only"]) == "only--"

    def test_star_alias(self) -> None:
        assert expand_template("$*", ["a", "b"]) == "a b"

    def test_non_recursive(self) -> None:
        # $1 contains "$2" literally — must NOT be re-expanded.
        assert expand_template("$1", ["$2 stays"]) == "$2 stays"


# ---------------------------------------------------------------------------
# Discovery from a directory
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discovers_markdown_files(self, tmp_path: Path) -> None:
        (tmp_path / "review.md").write_text(
            "---\nname: review\ndescription: Review a file\nargument-hint: <target>\n---\n"
            "Review the file at $1.\n",
            encoding="utf-8",
        )
        (tmp_path / "ignore.md").write_text("no frontmatter here", encoding="utf-8")

        ext = PromptTemplateExtension(templates=tmp_path)
        assert "review" in ext.templates
        assert ext.templates["review"].argument_hint == "<target>"

    def test_invalid_templates_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.md").write_text("---\nname: bad\n---\nno description\n", encoding="utf-8")
        ext = PromptTemplateExtension(templates=tmp_path)
        assert ext.templates == {}


# ---------------------------------------------------------------------------
# Programmatic render + tool
# ---------------------------------------------------------------------------


class TestRender:
    def _ext(self) -> PromptTemplateExtension:
        return PromptTemplateExtension(
            templates=[
                PromptTemplate(
                    name="hello",
                    description="Say hello",
                    body="Hello, $1!",
                    argument_hint="<name>",
                )
            ]
        )

    def test_render_with_string_args(self) -> None:
        out = self._ext().render("hello", '"World"')
        assert out == "Hello, World!"

    def test_render_with_sequence_args(self) -> None:
        out = self._ext().render("hello", ["Team"])
        assert out == "Hello, Team!"

    def test_render_unknown_raises(self) -> None:
        with pytest.raises(PromptTemplateError):
            self._ext().render("missing")

    def test_tool_registers(self) -> None:
        tools = self._ext().tools
        assert len(tools) == 1
        assert tools[0].name == "RunCommand"

    def test_prompt_roster_includes_name_and_description(self) -> None:
        out = self._ext().prompt({}, None)
        assert "`hello`" in out["prompt"]
        assert "Say hello" in out["prompt"]
        assert "<name>" in out["prompt"]

    @pytest.mark.asyncio
    async def test_tool_invokes_template(self) -> None:
        ext = self._ext()
        tool = ext.tools[0]
        out = await tool.ainvoke({"name": "hello", "args": "Alice"})
        assert out == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_tool_handles_unknown_template(self) -> None:
        ext = self._ext()
        tool = ext.tools[0]
        out = await tool.ainvoke({"name": "ghost", "args": ""})
        assert "Unknown template" in out
