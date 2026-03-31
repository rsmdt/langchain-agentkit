"""Tests for filesystem tools (Read, Write, Edit, Glob, Grep)."""

import tempfile
from pathlib import Path

from langchain_agentkit.backend import OSBackend
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools


def _make_backend_with_files(files: dict[str, str]) -> tuple[OSBackend, str]:
    """Create an OSBackend with pre-populated files. Returns (backend, tmpdir_path)."""
    tmpdir = tempfile.mkdtemp()
    backend = OSBackend(tmpdir)
    for path, content in files.items():
        backend.write(path, content)
    return backend, tmpdir


class TestCreateFilesystemTools:
    def test_returns_five_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tools = create_filesystem_tools(backend)

            assert len(tools) == 5

    def test_tool_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tools = create_filesystem_tools(backend)
            names = [t.name for t in tools]

            assert names == ["Read", "Write", "Edit", "Glob", "Grep"]

    def test_all_tools_have_descriptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tools = create_filesystem_tools(backend)

            for tool in tools:
                assert tool.description, f"{tool.name} has no description"

    def test_tools_share_same_backend(self):
        backend, tmpdir = _make_backend_with_files({})
        tools = create_filesystem_tools(backend)
        write_tool = tools[1]
        read_tool = tools[0]

        write_tool.invoke({"file_path": "/shared.txt", "content": "shared"})
        result = read_tool.invoke({"file_path": "/shared.txt"})

        assert "shared" in result


# --- Read Tool ---


class TestReadTool:
    def test_reads_file_with_line_numbers(self):
        backend, _ = _make_backend_with_files({"/test.txt": "line one\nline two\nline three"})
        tool = create_filesystem_tools(backend)[0]

        result = tool.invoke({"file_path": "/test.txt"})

        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    def test_offset_and_limit(self):
        backend, _ = _make_backend_with_files({"/test.txt": "a\nb\nc\nd\ne\n"})
        tool = create_filesystem_tools(backend)[0]

        result = tool.invoke({"file_path": "/test.txt", "offset": 1, "limit": 2})

        assert "2\tb" in result
        assert "3\tc" in result
        assert "a" not in result

    def test_unicode_content(self):
        backend, _ = _make_backend_with_files({"/f.txt": "日本語\n中文\nعربي"})
        tool = create_filesystem_tools(backend)[0]

        result = tool.invoke({"file_path": "/f.txt"})

        assert "日本語" in result
        assert "中文" in result
        assert "عربي" in result

    def test_single_line_file(self):
        backend, _ = _make_backend_with_files({"/f.txt": "only line\n"})
        tool = create_filesystem_tools(backend)[0]

        result = tool.invoke({"file_path": "/f.txt"})

        assert "1\tonly line" in result


# --- Write Tool ---


class TestWriteTool:
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]

            result = tool.invoke({"file_path": "/new.txt", "content": "hello"})

            assert "Wrote" in result
            assert (Path(tmpdir) / "new.txt").read_text() == "hello"

    def test_overwrites_existing(self):
        backend, tmpdir = _make_backend_with_files({"/f.txt": "old"})
        tool = create_filesystem_tools(backend)[1]

        tool.invoke({"file_path": "/f.txt", "content": "new"})

        assert (Path(tmpdir) / "f.txt").read_text() == "new"

    def test_creates_nested_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]

            tool.invoke({"file_path": "/a/b/c/deep.txt", "content": "deep"})

            assert (Path(tmpdir) / "a" / "b" / "c" / "deep.txt").read_text() == "deep"


# --- Edit Tool ---


class TestEditTool:
    def test_replaces_string(self):
        backend, tmpdir = _make_backend_with_files({"/f.txt": "hello world"})
        tool = create_filesystem_tools(backend)[2]

        result = tool.invoke(
            {
                "file_path": "/f.txt",
                "old_string": "hello",
                "new_string": "hi",
            }
        )

        assert "Replaced" in result
        assert (Path(tmpdir) / "f.txt").read_text() == "hi world"

    def test_replace_all(self):
        backend, tmpdir = _make_backend_with_files({"/f.txt": "foo bar foo baz foo"})
        tool = create_filesystem_tools(backend)[2]

        result = tool.invoke(
            {
                "file_path": "/f.txt",
                "old_string": "foo",
                "new_string": "qux",
                "replace_all": True,
            }
        )

        assert "3 occurrence(s)" in result
        assert (Path(tmpdir) / "f.txt").read_text() == "qux bar qux baz qux"

    def test_multiline_edit(self):
        backend, tmpdir = _make_backend_with_files({"/f.py": "def foo():\n    pass\n"})
        tool = create_filesystem_tools(backend)[2]

        tool.invoke(
            {
                "file_path": "/f.py",
                "old_string": "def foo():\n    pass",
                "new_string": "def foo():\n    return 42",
            }
        )

        assert (Path(tmpdir) / "f.py").read_text() == "def foo():\n    return 42\n"


# --- Glob Tool ---


class TestGlobTool:
    def test_finds_files(self):
        backend, _ = _make_backend_with_files(
            {
                "/a/b.md": "",
                "/a/c.txt": "",
            }
        )
        tool = create_filesystem_tools(backend)[3]

        result = tool.invoke({"pattern": "**/*.md"})

        assert "/a/b.md" in result
        assert "c.txt" not in result

    def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[3]

            result = tool.invoke({"pattern": "*.xyz"})

            assert "No files matched" in result

    def test_double_star_pattern(self):
        backend, _ = _make_backend_with_files(
            {
                "/skills/a/SKILL.md": "",
                "/skills/b/SKILL.md": "",
                "/skills/b/ref.py": "",
            }
        )
        tool = create_filesystem_tools(backend)[3]

        result = tool.invoke({"pattern": "**/*.md"})

        assert "SKILL.md" in result
        assert "ref.py" not in result


# --- Grep Tool ---


class TestGrepTool:
    """Tests for Grep tool — default output_mode is files_with_matches."""

    def test_default_mode_returns_file_paths(self):
        backend, _ = _make_backend_with_files({"/f.txt": "hello world\ngoodbye"})
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke({"pattern": "hello"})

        assert "/f.txt" in result

    def test_no_matches(self):
        backend, _ = _make_backend_with_files({"/f.txt": "nothing"})
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke({"pattern": "missing"})

        assert "No matches" in result

    def test_content_mode(self):
        backend, _ = _make_backend_with_files({"/f.txt": "hello world\ngoodbye"})
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke(
            {
                "pattern": "hello",
                "output_mode": "content",
            }
        )

        assert "/f.txt:1:" in result
        assert "hello world" in result

    def test_count_mode(self):
        backend, _ = _make_backend_with_files(
            {
                "/a.txt": "match\nmatch\nmatch",
                "/b.txt": "match",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke(
            {
                "pattern": "match",
                "output_mode": "count",
            }
        )

        assert "/a.txt: 3 match(es)" in result
        assert "/b.txt: 1 match(es)" in result

    def test_path_restriction(self):
        backend, _ = _make_backend_with_files(
            {
                "/a/file.txt": "target",
                "/b/file.txt": "target",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke({"pattern": "target", "path": "/a"})

        assert "/a/file.txt" in result
        assert "/b/file.txt" not in result

    def test_multiple_files_default_mode(self):
        backend, _ = _make_backend_with_files(
            {
                "/a.txt": "match here",
                "/b.txt": "match there",
                "/c.txt": "no hit",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke({"pattern": "match"})

        assert "/a.txt" in result
        assert "/b.txt" in result
        assert "/c.txt" not in result

    def test_head_limit_files_mode(self):
        backend, _ = _make_backend_with_files(
            {
                "/a.txt": "match",
                "/b.txt": "match",
                "/c.txt": "match",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke(
            {
                "pattern": "match",
                "head_limit": 2,
            }
        )

        lines = result.strip().split("\n")
        assert len([x for x in lines if x.startswith("/")]) == 2
        assert "1 more" in result

    def test_ignore_case(self):
        backend, _ = _make_backend_with_files({"/f.txt": "Hello World"})
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke(
            {
                "pattern": "hello",
                "ignore_case": True,
                "output_mode": "content",
            }
        )

        assert "Hello World" in result

    def test_case_sensitive_by_default(self):
        backend, _ = _make_backend_with_files({"/f.txt": "Hello World"})
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke({"pattern": "hello"})

        assert "No matches" in result

    def test_context_lines(self):
        backend, _ = _make_backend_with_files(
            {
                "/f.txt": "line1\nline2\nTARGET\nline4\nline5",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = tool.invoke(
            {
                "pattern": "TARGET",
                "output_mode": "content",
                "context": 1,
            }
        )

        assert "line2" in result
        assert "TARGET" in result
        assert "line4" in result


# --- LS Tool ---


class TestBashTool:
    """Test the Bash tool built by FilesystemExtension."""

    def test_bash_tool_runs_command(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)

            result = tool.invoke({"command": "echo hello"})

            assert "hello" in result

    def test_bash_tool_returns_exit_code_on_failure(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)

            result = tool.invoke({"command": "exit 1"})

            assert "Exit code 1" in result


# --- _strip_line_numbers ---


class TestStripLineNumbers:
    def test_strips_basic_line_numbers(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        formatted = "1\thello\n2\tworld\n"
        assert _strip_line_numbers(formatted) == "hello\nworld\n"

    def test_empty_input(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        assert _strip_line_numbers("") == ""

    def test_preserves_tabs_in_content(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        formatted = "1\tdata\twith\ttabs\n"
        assert _strip_line_numbers(formatted) == "data\twith\ttabs\n"

    def test_multi_digit_line_numbers(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        formatted = "100\tline hundred\n101\tline hundred one\n"
        assert _strip_line_numbers(formatted) == "line hundred\nline hundred one\n"

    def test_no_tab_in_line(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        # When no tab exists, partition returns ("no tab here\n", "", "")
        # Content becomes empty string — expected since valid
        # backend.read() output always has tabs
        formatted = "no tab here\n"
        result = _strip_line_numbers(formatted)
        assert result == ""

    def test_preserves_frontmatter(self):
        from langchain_agentkit.extensions.skills.discovery import _strip_line_numbers

        formatted = "1\t---\n2\tname: test\n3\tdescription: a test\n4\t---\n5\t# Body\n"
        result = _strip_line_numbers(formatted)
        assert result == "---\nname: test\ndescription: a test\n---\n# Body\n"
