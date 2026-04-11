"""Tests for filesystem tools (Read, Write, Edit, Glob, Grep)."""

import json
import tempfile
from pathlib import Path

from langchain_agentkit.backends import OSBackend
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools


async def _make_backend_with_files(files: dict[str, str]) -> tuple[OSBackend, str]:
    """Create an OSBackend with pre-populated files. Returns (backend, tmpdir_path)."""
    tmpdir = tempfile.mkdtemp()
    backend = OSBackend(tmpdir)
    for path, content in files.items():
        await backend.write(path, content)
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

    async def test_tools_share_same_backend(self):
        backend, tmpdir = await _make_backend_with_files({})
        tools = create_filesystem_tools(backend)
        write_tool = tools[1]
        read_tool = tools[0]

        await write_tool.ainvoke({"file_path": "/shared.txt", "content": "shared"})
        result = await read_tool.ainvoke({"file_path": "/shared.txt"})

        assert "shared" in result


# --- Read Tool ---


class TestReadTool:
    async def test_reads_file_with_line_numbers(self):
        backend, _ = await _make_backend_with_files({"/test.txt": "line one\nline two\nline three"})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/test.txt"})

        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    async def test_offset_and_limit(self):
        backend, _ = await _make_backend_with_files({"/test.txt": "a\nb\nc\nd\ne\n"})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/test.txt", "offset": 1, "limit": 2})

        assert "2\tb" in result
        assert "3\tc" in result
        assert "1\ta" not in result

    async def test_unicode_content(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "日本語\n中文\nعربي"})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/f.txt"})

        assert "日本語" in result
        assert "中文" in result
        assert "عربي" in result

    async def test_single_line_file(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "only line\n"})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/f.txt"})

        assert "1\tonly line" in result

    async def test_truncated_read_returns_partial_content(self):
        lines = "\n".join(f"line {i}" for i in range(100))
        backend, _ = await _make_backend_with_files({"/big.txt": lines})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/big.txt", "limit": 10})

        # Content has first 10 lines; totalLines available in artifact
        assert "line 0" in result
        assert "line 9" in result

    async def test_small_file_returns_all_content(self):
        backend, _ = await _make_backend_with_files({"/small.txt": "a\nb\nc\n"})
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/small.txt"})

        assert "total lines" not in result

    # --- Binary file rejection ---

    async def test_rejects_binary_extension(self):
        backend, tmpdir = await _make_backend_with_files({})
        await backend.write("/data.exe", b"\x00\x01\x02")
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/data.exe"})

        assert "binary" in result.lower()

    async def test_rejects_zip_file(self):
        backend, tmpdir = await _make_backend_with_files({})
        await backend.write("/archive.zip", b"PK\x03\x04")
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/archive.zip"})

        assert "binary" in result.lower()

    async def test_rejects_xlsx_file(self):
        backend, tmpdir = await _make_backend_with_files({})
        await backend.write("/sheet.xlsx", b"PK\x03\x04")
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/sheet.xlsx"})

        assert "binary" in result.lower()

    # --- Image carve-out ---

    async def test_reads_png_as_base64(self):
        backend, tmpdir = await _make_backend_with_files({})
        data = b"\x89PNG\r\n\x1a\n"
        await backend.write("/image.png", data)
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/image.png"})

        import base64

        assert base64.b64encode(data).decode() in result

    async def test_reads_jpg_as_base64(self):
        backend, tmpdir = await _make_backend_with_files({})
        data = b"\xff\xd8\xff\xe0"
        await backend.write("/photo.jpg", data)
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/photo.jpg"})

        import base64

        assert base64.b64encode(data).decode() in result

    async def test_reads_jpeg_as_base64(self):
        backend, tmpdir = await _make_backend_with_files({})
        data = b"\xff\xd8\xff\xe0"
        await backend.write("/photo.jpeg", data)
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/photo.jpeg"})

        import base64

        assert base64.b64encode(data).decode() in result

    # --- PDF carve-out ---

    async def test_reads_pdf_with_metadata_message(self):
        backend, tmpdir = await _make_backend_with_files({})
        data = b"%PDF-1.4 fake content"
        await backend.write("/doc.pdf", data)
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/doc.pdf"})

        assert "PDF file read:" in result
        assert "/doc.pdf" in result

    # --- Notebook carve-out ---

    async def test_reads_notebook_cells(self):
        import json

        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [
                        {"output_type": "stream", "text": ["hello\n"]},
                    ],
                    "metadata": {},
                },
                {
                    "cell_type": "markdown",
                    "source": ["# Title"],
                    "metadata": {},
                },
            ],
        }
        backend, tmpdir = await _make_backend_with_files({})
        await backend.write("/nb.ipynb", json.dumps(notebook))
        tool = create_filesystem_tools(backend)[0]

        result = await tool.ainvoke({"file_path": "/nb.ipynb"})

        assert "print('hello')" in result
        assert "hello" in result
        assert "# Title" in result


# --- Write Tool ---


class TestWriteTool:
    async def test_creates_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]

            result = await tool.ainvoke({"file_path": "/new.txt", "content": "hello"})

            assert "created successfully" in result
            assert (Path(tmpdir) / "new.txt").read_text() == "hello"

    async def test_updates_existing_file(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "old"})
        tool = create_filesystem_tools(backend)[1]

        result = await tool.ainvoke({"file_path": "/f.txt", "content": "new"})

        assert "has been updated successfully" in result
        assert (Path(tmpdir) / "f.txt").read_text() == "new"

    async def test_creates_nested_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]

            result = await tool.ainvoke({"file_path": "/a/b/c/deep.txt", "content": "deep"})

            assert "created successfully" in result
            assert (Path(tmpdir) / "a" / "b" / "c" / "deep.txt").read_text() == "deep"


# --- Edit Tool ---


class TestEditTool:
    async def test_replaces_string(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "hello world"})
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": "hello",
                "new_string": "hi",
            }
        )

        assert "has been updated successfully" in result
        assert (Path(tmpdir) / "f.txt").read_text() == "hi world"

    async def test_replace_all(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "foo bar foo baz foo"})
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": "foo",
                "new_string": "qux",
                "replace_all": True,
            }
        )

        assert "All occurrences were successfully replaced" in result
        assert (Path(tmpdir) / "f.txt").read_text() == "qux bar qux baz qux"

    async def test_multiline_edit(self):
        backend, tmpdir = await _make_backend_with_files({"/f.py": "def foo():\n    pass\n"})
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/f.py",
                "old_string": "def foo():\n    pass",
                "new_string": "def foo():\n    return 42",
            }
        )

        assert "has been updated successfully" in result
        assert (Path(tmpdir) / "f.py").read_text() == "def foo():\n    return 42\n"

    # --- Empty old_string for file creation ---

    async def test_empty_old_string_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[2]

            result = await tool.ainvoke(
                {
                    "file_path": "/new.txt",
                    "old_string": "",
                    "new_string": "hello world",
                }
            )

            assert "has been updated successfully" in result
            assert (Path(tmpdir) / "new.txt").read_text() == "hello world"

    async def test_empty_old_string_fills_empty_file(self):
        backend, tmpdir = await _make_backend_with_files({"/empty.txt": ""})
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/empty.txt",
                "old_string": "",
                "new_string": "content",
            }
        )

        assert "has been updated successfully" in result
        assert (Path(tmpdir) / "empty.txt").read_text() == "content"

    async def test_empty_old_string_rejects_nonempty_file(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "existing"})
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": "",
                "new_string": "new",
            }
        )

        assert "not empty" in result.lower() or "error" in result.lower()
        # File should not be modified
        assert (Path(tmpdir) / "f.txt").read_text() == "existing"

    # --- Trailing newline stripping on delete ---

    async def test_delete_strips_trailing_newline(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "line1\nline2\nline3\n"})
        tool = create_filesystem_tools(backend)[2]

        await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": "line2",
                "new_string": "",
            }
        )

        assert (Path(tmpdir) / "f.txt").read_text() == "line1\nline3\n"

    # --- Quote normalization ---

    async def test_curly_to_straight_quote_matching(self):
        backend, tmpdir = await _make_backend_with_files(
            {"/f.txt": "She said \u201chello\u201d to him"}
        )
        tool = create_filesystem_tools(backend)[2]

        result = await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": 'She said "hello" to him',
                "new_string": 'She said "goodbye" to him',
            }
        )

        assert "has been updated successfully" in result
        # preserveQuoteStyle converts straight quotes in new_string to curly
        text = (Path(tmpdir) / "f.txt").read_text()
        assert "\u201c" in text  # left curly
        assert "\u201d" in text  # right curly
        assert "goodbye" in text

    # --- Trailing whitespace stripping ---

    async def test_strips_trailing_whitespace_from_new_string(self):
        backend, tmpdir = await _make_backend_with_files({"/f.py": "x = 1\n"})
        tool = create_filesystem_tools(backend)[2]

        await tool.ainvoke(
            {
                "file_path": "/f.py",
                "old_string": "x = 1",
                "new_string": "x = 2   ",  # trailing spaces
            }
        )

        assert (Path(tmpdir) / "f.py").read_text() == "x = 2\n"

    async def test_preserves_trailing_whitespace_in_markdown(self):
        backend, tmpdir = await _make_backend_with_files({"/f.md": "line one\n"})
        tool = create_filesystem_tools(backend)[2]

        await tool.ainvoke(
            {
                "file_path": "/f.md",
                "old_string": "line one",
                "new_string": "line one  ",  # trailing spaces = hard break
            }
        )

        assert (Path(tmpdir) / "f.md").read_text() == "line one  \n"

    async def test_delete_preserves_when_old_string_ends_with_newline(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "line1\nline2\nline3\n"})
        tool = create_filesystem_tools(backend)[2]

        await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": "line2\n",
                "new_string": "",
            }
        )

        assert (Path(tmpdir) / "f.txt").read_text() == "line1\nline3\n"


# --- Glob Tool ---


class TestGlobTool:
    async def test_finds_files(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a/b.md": "",
                "/a/c.txt": "",
            }
        )
        tool = create_filesystem_tools(backend)[3]

        result = await tool.ainvoke({"pattern": "**/*.md"})

        assert "/a/b.md" in result
        assert "c.txt" not in result

    async def test_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[3]

            result = await tool.ainvoke({"pattern": "*.xyz"})

            assert "No files found" in result

    async def test_double_star_pattern(self):
        backend, _ = await _make_backend_with_files(
            {
                "/skills/a/SKILL.md": "",
                "/skills/b/SKILL.md": "",
                "/skills/b/ref.py": "",
            }
        )
        tool = create_filesystem_tools(backend)[3]

        result = await tool.ainvoke({"pattern": "**/*.md"})

        assert "SKILL.md" in result
        assert "ref.py" not in result


# --- Grep Tool ---


class TestGrepTool:
    """Tests for Grep tool — default output_mode is files_with_matches."""

    async def test_default_mode_returns_file_paths(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello world\ngoodbye"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "hello"})

        assert "/f.txt" in result

    async def test_no_matches(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "nothing"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "missing"})

        assert "No matches" in result

    async def test_content_mode(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello world\ngoodbye"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "hello",
                "output_mode": "content",
            }
        )

        assert "/f.txt:1:" in result
        assert "hello world" in result

    async def test_count_mode(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a.txt": "match\nmatch\nmatch",
                "/b.txt": "match",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "match",
                "output_mode": "count",
            }
        )

        assert "/a.txt: 3 match(es)" in result
        assert "/b.txt: 1 match(es)" in result

    async def test_path_restriction(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a/file.txt": "target",
                "/b/file.txt": "target",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "target", "path": "/a"})

        assert "/a/file.txt" in result
        assert "/b/file.txt" not in result

    async def test_multiple_files_default_mode(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a.txt": "match here",
                "/b.txt": "match there",
                "/c.txt": "no hit",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "match"})

        assert "/a.txt" in result
        assert "/b.txt" in result
        assert "/c.txt" not in result

    async def test_head_limit_files_mode(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a.txt": "match",
                "/b.txt": "match",
                "/c.txt": "match",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "match",
                "head_limit": 2,
            }
        )

        lines = result.strip().split("\n")
        assert len([x for x in lines if x.startswith("/")]) == 2
        assert "use offset to paginate" in result

    async def test_ignore_case(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "Hello World"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "hello",
                "ignore_case": True,
                "output_mode": "content",
            }
        )

        assert "Hello World" in result

    async def test_case_sensitive_by_default(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "Hello World"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "hello"})

        assert "No matches" in result

    async def test_context_lines(self):
        backend, _ = await _make_backend_with_files(
            {
                "/f.txt": "line1\nline2\nTARGET\nline4\nline5",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "TARGET",
                "output_mode": "content",
                "context": 1,
            }
        )

        assert "line2" in result
        assert "TARGET" in result
        assert "line4" in result

    async def test_asymmetric_context_before_only(self):
        content = "line1\nline2\nTARGET\nline4\nline5"
        backend, _ = await _make_backend_with_files({"/f.txt": content})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "TARGET",
                "output_mode": "content",
                "-B": 1,
            }
        )

        assert "line2" in result
        assert "TARGET" in result
        assert "line4" not in result

    async def test_asymmetric_context_after_only(self):
        content = "line1\nline2\nTARGET\nline4\nline5"
        backend, _ = await _make_backend_with_files({"/f.txt": content})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "TARGET",
                "output_mode": "content",
                "-A": 1,
            }
        )

        assert "line2" not in result
        assert "TARGET" in result
        assert "line4" in result

    async def test_multiline_match(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "start\nhello\nworld\nend"})
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "hello.*world",
                "output_mode": "content",
                "multiline": True,
            }
        )

        assert "hello" in result
        assert "world" in result

    async def test_default_head_limit_is_250(self):
        """Default head_limit should be 250 per reference."""
        files = {f"/{i}.txt": "match" for i in range(300)}
        backend, _ = await _make_backend_with_files(files)
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "match"})

        lines = [x for x in result.strip().split("\n") if x.startswith("/")]
        assert len(lines) == 250

    async def test_offset_skips_entries(self):
        backend, _ = await _make_backend_with_files(
            {
                "/a.txt": "match",
                "/b.txt": "match",
                "/c.txt": "match",
            }
        )
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke(
            {
                "pattern": "match",
                "offset": 1,
                "head_limit": 1,
            }
        )

        lines = [x for x in result.strip().split("\n") if x.startswith("/")]
        assert len(lines) == 1

    async def test_bash_tool_accepts_description(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)

            result = await tool.ainvoke({"command": "echo hi", "description": "Print greeting"})

            assert "hi" in result


# --- Structured Output (artifacts) ---


class TestStructuredOutput:
    """Verify tools return structured artifacts via content_and_artifact."""

    async def _call_raw(self, tool, input_dict):
        """Call the inner coroutine directly to get (content, artifact) tuple."""
        coroutine = tool.coroutine
        return await coroutine(**input_dict)

    async def test_read_text_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello\nworld\n"})
        tool = create_filesystem_tools(backend)[0]
        content, artifact = await self._call_raw(tool, {"file_path": "/f.txt"})

        assert "hello" in content
        assert artifact["type"] == "text"
        assert artifact["filePath"] == "/f.txt"
        assert artifact["numLines"] == 2
        assert artifact["startLine"] == 1
        assert artifact["totalLines"] == 2
        assert "hello" in artifact["content"]

    async def test_read_image_artifact(self):
        backend, _ = await _make_backend_with_files({})
        await backend.write("/img.png", b"\x89PNG\r\n\x1a\n")
        tool = create_filesystem_tools(backend)[0]
        content, artifact = await self._call_raw(tool, {"file_path": "/img.png"})

        assert artifact["type"] == "image"
        assert artifact["mediaType"] == "image/png"
        assert artifact["originalSize"] == 8
        assert len(artifact["base64"]) > 0

    async def test_read_pdf_artifact(self):
        backend, _ = await _make_backend_with_files({})
        await backend.write("/doc.pdf", b"%PDF-1.4")
        tool = create_filesystem_tools(backend)[0]
        content, artifact = await self._call_raw(tool, {"file_path": "/doc.pdf"})

        assert artifact["type"] == "pdf"
        assert artifact["originalSize"] == 8
        assert len(artifact["base64"]) > 0

    async def test_read_notebook_artifact(self):
        import json as json_mod

        nb = {
            "nbformat": 4,
            "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["x = 1"], "outputs": [], "metadata": {}},
            ],
        }
        backend, _ = await _make_backend_with_files({})
        await backend.write("/nb.ipynb", json_mod.dumps(nb))
        tool = create_filesystem_tools(backend)[0]
        content, artifact = await self._call_raw(tool, {"file_path": "/nb.ipynb"})

        assert artifact["type"] == "notebook"
        assert isinstance(artifact["cells"], list)
        assert artifact["cells"][0]["source"] == "x = 1"

    async def test_write_create_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]
            content, artifact = await self._call_raw(
                tool, {"file_path": "/new.txt", "content": "hi"}
            )

            assert artifact["type"] == "create"
            assert artifact["filePath"] == "/new.txt"
            assert artifact["content"] == "hi"
            assert artifact["originalFile"] is None

    async def test_write_update_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "old"})
        tool = create_filesystem_tools(backend)[1]
        content, artifact = await self._call_raw(tool, {"file_path": "/f.txt", "content": "new"})

        assert artifact["type"] == "update"
        assert artifact["content"] == "new"
        assert artifact["originalFile"] == "old"

    async def test_edit_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello world"})
        tool = create_filesystem_tools(backend)[2]
        content, artifact = await self._call_raw(
            tool,
            {"file_path": "/f.txt", "old_string": "hello", "new_string": "hi"},
        )

        assert artifact["filePath"] == "/f.txt"
        assert artifact["oldString"] == "hello"
        assert artifact["newString"] == "hi"
        assert artifact["replaceAll"] is False
        assert artifact["originalFile"] == "hello world"

    async def test_glob_artifact(self):
        backend, _ = await _make_backend_with_files({"/a.txt": "", "/b.txt": ""})
        tool = create_filesystem_tools(backend)[3]
        content, artifact = await self._call_raw(tool, {"pattern": "**/*.txt"})

        assert artifact["numFiles"] == 2
        assert len(artifact["filenames"]) == 2
        assert artifact["truncated"] is False
        assert "durationMs" in artifact

    async def test_grep_files_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "match here"})
        tool = create_filesystem_tools(backend)[4]
        content, artifact = await self._call_raw(tool, {"pattern": "match"})

        assert artifact["mode"] == "files_with_matches"
        assert artifact["numFiles"] == 1
        assert "/f.txt" in artifact["filenames"]

    async def test_grep_content_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "match\nno\nmatch"})
        tool = create_filesystem_tools(backend)[4]
        content, artifact = await self._call_raw(
            tool,
            {"pattern": "match", "output_mode": "content"},
        )

        assert artifact["mode"] == "content"
        assert artifact["numLines"] == 2

    async def test_grep_count_artifact(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "a\na\na"})
        tool = create_filesystem_tools(backend)[4]
        content, artifact = await self._call_raw(
            tool,
            {"pattern": "a", "output_mode": "count"},
        )

        assert artifact["mode"] == "count"
        assert artifact["numMatches"] == 3

    async def test_bash_artifact(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)
            content, artifact = await tool.coroutine(command="echo hello")

            assert artifact["stdout"].strip() == "hello"
            assert artifact["stderr"] == ""
            assert artifact["exitCode"] == 0
            assert artifact["interrupted"] is False

    async def test_write_update_has_structured_patch(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello\nworld\n"})
        tool = create_filesystem_tools(backend)[1]
        _, artifact = await self._call_raw(
            tool, {"file_path": "/f.txt", "content": "hello\nplanet\n"}
        )

        assert "structuredPatch" in artifact
        assert len(artifact["structuredPatch"]) > 0
        hunk = artifact["structuredPatch"][0]
        assert "oldStart" in hunk
        assert "newStart" in hunk
        assert "lines" in hunk
        assert any("+planet" in line for line in hunk["lines"])

    async def test_write_create_has_empty_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = create_filesystem_tools(backend)[1]
            _, artifact = await self._call_raw(tool, {"file_path": "/new.txt", "content": "hi"})

            assert artifact["structuredPatch"] == []

    async def test_edit_has_structured_patch(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello world"})
        tool = create_filesystem_tools(backend)[2]
        _, artifact = await self._call_raw(
            tool,
            {"file_path": "/f.txt", "old_string": "hello", "new_string": "hi"},
        )

        assert "structuredPatch" in artifact
        assert len(artifact["structuredPatch"]) > 0

    async def test_edit_preserves_curly_quotes_in_new_string(self):
        backend, tmpdir = await _make_backend_with_files({"/f.txt": "She said \u201chello\u201d"})
        tool = create_filesystem_tools(backend)[2]
        await tool.ainvoke(
            {
                "file_path": "/f.txt",
                "old_string": 'She said "hello"',
                "new_string": 'She said "goodbye"',
            }
        )
        result = (Path(tmpdir) / "f.txt").read_text()
        # new_string should have curly quotes preserved
        assert "\u201c" in result  # left double curly
        assert "\u201d" in result  # right double curly

    async def test_read_text_file_unchanged_on_second_read(self):
        backend, _ = await _make_backend_with_files({"/f.txt": "hello\n"})
        tool = create_filesystem_tools(backend)[0]
        # First read populates cache
        content1, art1 = await self._call_raw(tool, {"file_path": "/f.txt"})
        assert art1["type"] == "text"
        # Second read with same params returns file_unchanged
        content2, art2 = await self._call_raw(tool, {"file_path": "/f.txt"})
        assert art2["type"] == "file_unchanged"
        assert "not changed" in content2.lower()

    async def test_read_image_has_multimodal_content(self):
        backend, _ = await _make_backend_with_files({})
        await backend.write("/img.png", b"\x89PNG\r\n\x1a\n")
        tool = create_filesystem_tools(backend)[0]
        content, _ = await self._call_raw(tool, {"file_path": "/img.png"})
        # Content should be JSON with image block structure
        parsed = json.loads(content)
        assert isinstance(parsed, list)
        assert parsed[0]["type"] == "image"
        assert parsed[0]["source"]["type"] == "base64"
        assert parsed[0]["source"]["media_type"] == "image/png"

    async def test_read_pdf_message_format(self):
        backend, _ = await _make_backend_with_files({})
        await backend.write("/doc.pdf", b"%PDF-1.4")
        tool = create_filesystem_tools(backend)[0]
        content, _ = await self._call_raw(tool, {"file_path": "/doc.pdf"})
        assert content.startswith("PDF file read:")
        assert "/doc.pdf" in content

    async def test_read_empty_file_warning(self):
        backend, _ = await _make_backend_with_files({"/empty.txt": ""})
        tool = create_filesystem_tools(backend)[0]
        content, artifact = await self._call_raw(tool, {"file_path": "/empty.txt"})
        assert "empty" in content.lower()
        assert artifact["totalLines"] == 0

    async def test_grep_files_with_matches_shows_found_count(self):
        backend, _ = await _make_backend_with_files({"/a.txt": "x", "/b.txt": "x"})
        tool = create_filesystem_tools(backend)[4]
        result = await tool.ainvoke({"pattern": "x"})

        assert "Found 2 file(s)" in result

    async def test_grep_truncated_shows_pagination(self):
        files = {f"/{i}.txt": "match" for i in range(300)}
        backend, _ = await _make_backend_with_files(files)
        tool = create_filesystem_tools(backend)[4]

        result = await tool.ainvoke({"pattern": "match"})

        assert "use offset to paginate" in result


# --- Bash Tool ---


class TestBashTool:
    """Test the Bash tool built by FilesystemExtension."""

    async def test_bash_tool_runs_command(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)

            result = await tool.ainvoke({"command": "echo hello"})

            assert "hello" in result

    async def test_bash_tool_returns_exit_code_on_failure(self):
        from langchain_agentkit.extensions.filesystem.extension import _build_bash_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OSBackend(tmpdir)
            tool = _build_bash_tool(backend)

            result = await tool.ainvoke({"command": "exit 1"})

            assert "Exit code 1" in result
