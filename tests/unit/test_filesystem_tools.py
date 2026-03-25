"""Tests for filesystem tools (Read, Write, Edit, Glob, Grep)."""


from langchain_agentkit.filesystem_tools import (
    _format_with_line_numbers,
    create_filesystem_tools,
)
from langchain_agentkit.virtual_filesystem import VirtualFilesystem


class TestCreateFilesystemTools:
    def test_returns_five_tools(self):
        vfs = VirtualFilesystem()
        tools = create_filesystem_tools(vfs)

        assert len(tools) == 5

    def test_tool_names(self):
        vfs = VirtualFilesystem()
        tools = create_filesystem_tools(vfs)
        names = [t.name for t in tools]

        assert names == ["Read", "Write", "Edit", "Glob", "Grep"]

    def test_all_tools_have_descriptions(self):
        vfs = VirtualFilesystem()
        tools = create_filesystem_tools(vfs)

        for tool in tools:
            assert tool.description, f"{tool.name} has no description"

    def test_tools_share_same_vfs(self):
        vfs = VirtualFilesystem()
        tools = create_filesystem_tools(vfs)
        write_tool = tools[1]
        read_tool = tools[0]

        write_tool.invoke({"file_path": "/shared.txt", "content": "shared"})
        result = read_tool.invoke({"file_path": "/shared.txt"})

        assert "shared" in result


# --- Format With Line Numbers ---


class TestFormatWithLineNumbers:
    def test_basic_formatting(self):
        result = _format_with_line_numbers("a\nb\nc", 0, 2000)

        assert "1\ta" in result
        assert "2\tb" in result
        assert "3\tc" in result

    def test_offset(self):
        result = _format_with_line_numbers("a\nb\nc\nd", 1, 2000)

        assert "2\tb" in result
        assert "1\ta" not in result

    def test_limit(self):
        result = _format_with_line_numbers("a\nb\nc\nd\ne", 0, 2)

        assert "1\ta" in result
        assert "2\tb" in result
        assert "3\tc" not in result

    def test_shows_remaining_count(self):
        result = _format_with_line_numbers("a\nb\nc\nd\ne", 0, 2)

        assert "3 more lines" in result

    def test_no_remaining_when_all_shown(self):
        result = _format_with_line_numbers("a\nb", 0, 2000)

        assert "more lines" not in result

    def test_offset_beyond_content(self):
        result = _format_with_line_numbers("a\nb", 10, 2000)

        assert result == ""

    def test_line_number_width_alignment(self):
        lines = "\n".join(f"line{i}" for i in range(100))
        result = _format_with_line_numbers(lines, 0, 100)

        # Line 1 should be padded to match width of "100"
        assert "  1\tline0" in result
        assert "100\tline99" in result


# --- Read Tool ---


class TestReadTool:
    def test_reads_file_with_line_numbers(self):
        vfs = VirtualFilesystem()
        vfs.write("/test.txt", "line one\nline two\nline three")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/test.txt"})

        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    def test_offset_and_limit(self):
        vfs = VirtualFilesystem()
        vfs.write("/test.txt", "a\nb\nc\nd\ne")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/test.txt", "offset": 1, "limit": 2})

        assert "2\tb" in result
        assert "3\tc" in result
        assert "a" not in result

    def test_file_not_found(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/nope.txt"})

        assert "not found" in result.lower()

    def test_empty_file(self):
        vfs = VirtualFilesystem()
        vfs.write("/empty.txt", "")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/empty.txt"})

        assert "empty" in result.lower()

    def test_large_file_with_limit(self):
        vfs = VirtualFilesystem()
        content = "\n".join(f"line {i}" for i in range(1000))
        vfs.write("/large.txt", content)
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/large.txt", "limit": 5})

        assert "line 0" in result
        assert "line 4" in result
        assert "995 more lines" in result

    def test_default_offset_is_zero(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "first\nsecond")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/f.txt"})

        assert "1\tfirst" in result

    def test_unicode_content(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "日本語\n中文\nعربي")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/f.txt"})

        assert "日本語" in result
        assert "中文" in result
        assert "عربي" in result

    def test_single_line_file(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "only line")
        tool = create_filesystem_tools(vfs)[0]

        result = tool.invoke({"file_path": "/f.txt"})

        assert "1\tonly line" in result


# --- Write Tool ---


class TestWriteTool:
    def test_writes_file(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[1]

        result = tool.invoke({"file_path": "/new.txt", "content": "hello"})

        assert "Wrote" in result
        assert vfs.read("/new.txt") == "hello"

    def test_overwrites_existing(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "old")
        tool = create_filesystem_tools(vfs)[1]

        tool.invoke({"file_path": "/f.txt", "content": "new"})

        assert vfs.read("/f.txt") == "new"

    def test_reports_character_count(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[1]

        result = tool.invoke({"file_path": "/f.txt", "content": "12345"})

        assert "5 characters" in result

    def test_creates_nested_path(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[1]

        tool.invoke({"file_path": "/a/b/c/deep.txt", "content": "deep"})

        assert vfs.read("/a/b/c/deep.txt") == "deep"

    def test_write_empty_content(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[1]

        result = tool.invoke({"file_path": "/f.txt", "content": ""})

        assert "0 characters" in result
        assert vfs.read("/f.txt") == ""


# --- Edit Tool ---


class TestEditTool:
    def test_replaces_string(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello world")
        tool = create_filesystem_tools(vfs)[2]

        result = tool.invoke({
            "file_path": "/f.txt",
            "old_string": "hello",
            "new_string": "hi",
        })

        assert "Replaced" in result
        assert vfs.read("/f.txt") == "hi world"

    def test_file_not_found(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[2]

        result = tool.invoke({
            "file_path": "/nope.txt",
            "old_string": "a",
            "new_string": "b",
        })

        assert "not found" in result.lower()

    def test_string_not_found(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello")
        tool = create_filesystem_tools(vfs)[2]

        result = tool.invoke({
            "file_path": "/f.txt",
            "old_string": "missing",
            "new_string": "x",
        })

        assert "not found" in result.lower()

    def test_replace_all(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "foo bar foo baz foo")
        tool = create_filesystem_tools(vfs)[2]

        result = tool.invoke({
            "file_path": "/f.txt",
            "old_string": "foo",
            "new_string": "qux",
            "replace_all": True,
        })

        assert "3 occurrence(s)" in result
        assert vfs.read("/f.txt") == "qux bar qux baz qux"

    def test_ambiguous_without_replace_all(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "aa")
        tool = create_filesystem_tools(vfs)[2]

        result = tool.invoke({
            "file_path": "/f.txt",
            "old_string": "a",
            "new_string": "b",
        })

        # Should return error via handle_tool_error
        assert "occurrences" in result.lower() or "2" in result

    def test_multiline_edit(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.py", "def foo():\n    pass\n")
        tool = create_filesystem_tools(vfs)[2]

        tool.invoke({
            "file_path": "/f.py",
            "old_string": "def foo():\n    pass",
            "new_string": "def foo():\n    return 42",
        })

        assert vfs.read("/f.py") == "def foo():\n    return 42\n"


# --- Glob Tool ---


class TestGlobTool:
    def test_finds_files(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b.md", "")
        vfs.write("/a/c.txt", "")
        tool = create_filesystem_tools(vfs)[3]

        result = tool.invoke({"pattern": "/a/*.md"})

        assert "/a/b.md" in result
        assert "/a/c.txt" not in result

    def test_no_matches(self):
        vfs = VirtualFilesystem()
        tool = create_filesystem_tools(vfs)[3]

        result = tool.invoke({"pattern": "/*.xyz"})

        assert "No files matched" in result

    def test_double_star_pattern(self):
        vfs = VirtualFilesystem()
        vfs.write("/skills/a/SKILL.md", "")
        vfs.write("/skills/b/SKILL.md", "")
        vfs.write("/skills/b/ref.py", "")
        tool = create_filesystem_tools(vfs)[3]

        result = tool.invoke({"pattern": "/skills/**/*.md"})

        assert "/skills/a/SKILL.md" in result
        assert "/skills/b/SKILL.md" in result
        assert "ref.py" not in result

    def test_results_are_sorted(self):
        vfs = VirtualFilesystem()
        vfs.write("/z.txt", "")
        vfs.write("/a.txt", "")
        tool = create_filesystem_tools(vfs)[3]

        result = tool.invoke({"pattern": "/*.txt"})
        lines = result.strip().split("\n")

        assert lines[0] == "/a.txt"
        assert lines[1] == "/z.txt"


# --- Grep Tool ---


class TestGrepTool:
    def test_finds_matches(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello world\ngoodbye")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "hello"})

        assert "/f.txt:1:" in result
        assert "hello world" in result

    def test_no_matches(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "nothing")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "missing"})

        assert "No matches" in result

    def test_path_restriction(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/file.txt", "target")
        vfs.write("/b/file.txt", "target")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "target", "path": "/a"})

        assert "/a/file.txt" in result
        assert "/b/file.txt" not in result

    def test_case_insensitive(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "Hello World")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "hello", "ignore_case": True})

        assert "Hello World" in result

    def test_glob_filter(self):
        vfs = VirtualFilesystem()
        vfs.write("/d/a.py", "target")
        vfs.write("/d/b.md", "target")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "target", "glob": "/d/*.py"})

        assert "/d/a.py" in result
        assert "/d/b.md" not in result

    def test_regex_pattern(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "foo123\nbar456\nbaz")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": r"\d+"})

        assert "foo123" in result
        assert "bar456" in result
        assert "baz" not in result

    def test_multiple_files(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "match here")
        vfs.write("/b.txt", "match there")
        vfs.write("/c.txt", "no hit")
        tool = create_filesystem_tools(vfs)[4]

        result = tool.invoke({"pattern": "match"})

        assert "/a.txt:1:" in result
        assert "/b.txt:1:" in result
        assert "/c.txt" not in result
