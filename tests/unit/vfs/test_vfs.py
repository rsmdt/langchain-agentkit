"""Tests for VirtualFilesystem."""

from pathlib import Path

import pytest

from langchain_agentkit.vfs import VirtualFilesystem

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"

# --- Path Normalization ---


class TestNormalizePath:
    def test_adds_leading_slash(self):
        assert VirtualFilesystem.normalize_path("foo/bar") == "/foo/bar"

    def test_preserves_leading_slash(self):
        assert VirtualFilesystem.normalize_path("/foo/bar") == "/foo/bar"

    def test_resolves_dot_dot(self):
        assert VirtualFilesystem.normalize_path("/foo/bar/../baz") == "/foo/baz"

    def test_resolves_dot(self):
        assert VirtualFilesystem.normalize_path("/foo/./bar") == "/foo/bar"

    def test_root_path(self):
        assert VirtualFilesystem.normalize_path("/") == "/"

    def test_multiple_dot_dot(self):
        assert VirtualFilesystem.normalize_path("/a/b/c/../../d") == "/a/d"

    def test_dot_dot_at_root_stays_at_root(self):
        assert VirtualFilesystem.normalize_path("/a/../..") == "/"

    def test_trailing_slash_stripped(self):
        assert VirtualFilesystem.normalize_path("/foo/bar/") == "/foo/bar"

    def test_double_slash_collapsed(self):
        assert VirtualFilesystem.normalize_path("/foo//bar") == "/foo/bar"

    def test_deeply_nested_path(self):
        assert VirtualFilesystem.normalize_path("/a/b/c/d/e") == "/a/b/c/d/e"

    def test_empty_string_becomes_root(self):
        assert VirtualFilesystem.normalize_path("") == "/"

    def test_dot_only_becomes_root(self):
        assert VirtualFilesystem.normalize_path(".") == "/"

    def test_relative_path_with_dot(self):
        assert VirtualFilesystem.normalize_path("./foo/bar") == "/foo/bar"


# --- Write and Read ---


class TestWriteAndRead:
    def test_write_and_read(self):
        vfs = VirtualFilesystem()
        vfs.write("/test.txt", "hello")

        assert vfs.read("/test.txt") == "hello"

    def test_read_nonexistent_returns_none(self):
        vfs = VirtualFilesystem()

        assert vfs.read("/nope.txt") is None

    def test_write_overwrites(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "first")
        vfs.write("/f.txt", "second")

        assert vfs.read("/f.txt") == "second"

    def test_normalizes_path_on_write(self):
        vfs = VirtualFilesystem()
        vfs.write("foo/bar.txt", "content")

        assert vfs.read("/foo/bar.txt") == "content"

    def test_write_empty_string(self):
        vfs = VirtualFilesystem()
        vfs.write("/empty.txt", "")

        assert vfs.read("/empty.txt") == ""

    def test_write_multiline_content(self):
        vfs = VirtualFilesystem()
        content = "line1\nline2\nline3"
        vfs.write("/multi.txt", content)

        assert vfs.read("/multi.txt") == content

    def test_write_unicode_content(self):
        vfs = VirtualFilesystem()
        content = "Hello 🌍 日本語 العربية"
        vfs.write("/unicode.txt", content)

        assert vfs.read("/unicode.txt") == content

    def test_write_large_content(self):
        vfs = VirtualFilesystem()
        content = "x" * 100_000
        vfs.write("/large.txt", content)

        assert vfs.read("/large.txt") == content
        assert len(vfs.read("/large.txt")) == 100_000

    def test_write_content_with_special_characters(self):
        vfs = VirtualFilesystem()
        content = "tabs\there\nnulls\x00and\r\nwindows"
        vfs.write("/special.txt", content)

        assert vfs.read("/special.txt") == content

    def test_write_preserves_trailing_newline(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "content\n")

        assert vfs.read("/f.txt") == "content\n"

    def test_read_normalizes_path(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b/c.txt", "data")

        assert vfs.read("a/b/c.txt") == "data"
        assert vfs.read("/a/./b/c.txt") == "data"
        assert vfs.read("/a/b/../b/c.txt") == "data"


# --- Exists ---


class TestExists:
    def test_exists_true(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "x")

        assert vfs.exists("/a.txt")

    def test_exists_false(self):
        vfs = VirtualFilesystem()

        assert not vfs.exists("/nope.txt")

    def test_contains_protocol(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "x")

        assert "/a.txt" in vfs
        assert "/b.txt" not in vfs

    def test_exists_normalizes_path(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b.txt", "x")

        assert vfs.exists("a/b.txt")
        assert vfs.exists("/a/./b.txt")


# --- Delete ---


class TestDelete:
    def test_delete_existing(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "x")

        assert vfs.delete("/a.txt") is True
        assert not vfs.exists("/a.txt")

    def test_delete_nonexistent(self):
        vfs = VirtualFilesystem()

        assert vfs.delete("/nope.txt") is False

    def test_delete_normalizes_path(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b.txt", "x")

        assert vfs.delete("a/b.txt") is True
        assert not vfs.exists("/a/b.txt")


# --- List Directory ---


class TestListDirectory:
    def test_lists_files_and_dirs(self):
        vfs = VirtualFilesystem()
        vfs.write("/root/file.txt", "a")
        vfs.write("/root/sub/nested.txt", "b")

        children = vfs.list_directory("/root")

        assert "file.txt" in children
        assert "sub/" in children

    def test_empty_directory(self):
        vfs = VirtualFilesystem()

        assert vfs.list_directory("/empty") == []

    def test_sorted_output(self):
        vfs = VirtualFilesystem()
        vfs.write("/d/c.txt", "")
        vfs.write("/d/a.txt", "")
        vfs.write("/d/b.txt", "")

        assert vfs.list_directory("/d") == ["a.txt", "b.txt", "c.txt"]

    def test_deeply_nested_only_shows_immediate_children(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b/c/d.txt", "")
        vfs.write("/a/file.txt", "")

        children = vfs.list_directory("/a")

        assert children == ["b/", "file.txt"]

    def test_root_directory(self):
        vfs = VirtualFilesystem()
        vfs.write("/foo.txt", "")
        vfs.write("/bar/baz.txt", "")

        children = vfs.list_directory("/")

        assert "foo.txt" in children
        assert "bar/" in children

    def test_no_duplicate_dir_entries(self):
        vfs = VirtualFilesystem()
        vfs.write("/d/sub/a.txt", "")
        vfs.write("/d/sub/b.txt", "")

        children = vfs.list_directory("/d")

        assert children.count("sub/") == 1


# --- Glob ---


class TestGlob:
    def test_star_pattern(self):
        vfs = VirtualFilesystem()
        vfs.write("/skills/a/SKILL.md", "a")
        vfs.write("/skills/b/SKILL.md", "b")
        vfs.write("/skills/b/helper.py", "c")

        matches = vfs.glob("/skills/*/SKILL.md")

        assert matches == ["/skills/a/SKILL.md", "/skills/b/SKILL.md"]

    def test_double_star(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b/c.md", "")
        vfs.write("/a/d.md", "")

        matches = vfs.glob("/a/**/*.md")

        assert "/a/b/c.md" in matches
        assert "/a/d.md" in matches

    def test_no_matches(self):
        vfs = VirtualFilesystem()

        assert vfs.glob("/nothing/*.txt") == []

    def test_question_mark_wildcard(self):
        vfs = VirtualFilesystem()
        vfs.write("/a1.txt", "")
        vfs.write("/a2.txt", "")
        vfs.write("/ab.txt", "")

        matches = vfs.glob("/a?.txt")

        assert "/a1.txt" in matches
        assert "/a2.txt" in matches
        assert "/ab.txt" in matches

    def test_extension_filter(self):
        vfs = VirtualFilesystem()
        vfs.write("/dir/a.py", "")
        vfs.write("/dir/b.md", "")
        vfs.write("/dir/c.py", "")

        matches = vfs.glob("/dir/*.py")

        assert matches == ["/dir/a.py", "/dir/c.py"]

    def test_matches_are_sorted(self):
        vfs = VirtualFilesystem()
        vfs.write("/z.txt", "")
        vfs.write("/a.txt", "")
        vfs.write("/m.txt", "")

        assert vfs.glob("/*.txt") == ["/a.txt", "/m.txt", "/z.txt"]

    def test_double_star_deeply_nested(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b/c/d/e.txt", "")
        vfs.write("/a/f.txt", "")

        matches = vfs.glob("/a/**/*.txt")

        assert "/a/b/c/d/e.txt" in matches
        assert "/a/f.txt" in matches

    def test_star_does_not_cross_directory_boundary(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b.txt", "")
        vfs.write("/a/c/d.txt", "")

        matches = vfs.glob("/a/*.txt")

        assert matches == ["/a/b.txt"]


# --- Grep ---


class TestGrep:
    def test_finds_matching_lines(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "hello world\ngoodbye world\nhello again")

        results = vfs.grep("hello")

        assert len(results) == 2
        assert results[0]["line"] == 1
        assert results[0]["text"] == "hello world"
        assert results[1]["line"] == 3
        assert results[1]["text"] == "hello again"

    def test_path_restriction(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/file.txt", "match")
        vfs.write("/b/file.txt", "match")

        results = vfs.grep("match", path="/a")

        assert len(results) == 1
        assert results[0]["path"] == "/a/file.txt"

    def test_case_insensitive(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "Hello World")

        results = vfs.grep("hello", ignore_case=True)

        assert len(results) == 1

    def test_case_sensitive_by_default(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "Hello World")

        results = vfs.grep("hello", ignore_case=False)

        assert len(results) == 0

    def test_no_matches(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "nothing here")

        assert vfs.grep("missing") == []

    def test_invalid_regex_returns_empty(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "content")

        assert vfs.grep("[invalid") == []

    def test_regex_pattern(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "foo123\nbar456\nfoo789")

        results = vfs.grep(r"foo\d+")

        assert len(results) == 2

    def test_glob_filter(self):
        vfs = VirtualFilesystem()
        vfs.write("/dir/a.py", "match")
        vfs.write("/dir/b.md", "match")

        results = vfs.grep("match", glob_filter="/dir/*.py")

        assert len(results) == 1
        assert results[0]["path"] == "/dir/a.py"

    def test_results_sorted_by_file(self):
        vfs = VirtualFilesystem()
        vfs.write("/z.txt", "match")
        vfs.write("/a.txt", "match")

        results = vfs.grep("match")

        assert results[0]["path"] == "/a.txt"
        assert results[1]["path"] == "/z.txt"

    def test_empty_filesystem(self):
        vfs = VirtualFilesystem()

        assert vfs.grep("anything") == []

    def test_multiline_grep(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "line1 target\nline2\nline3 target\nline4")

        results = vfs.grep("target")

        assert len(results) == 2
        assert results[0]["line"] == 1
        assert results[1]["line"] == 3

    def test_grep_with_path_and_glob(self):
        vfs = VirtualFilesystem()
        vfs.write("/src/a.py", "match")
        vfs.write("/src/b.md", "match")
        vfs.write("/lib/c.py", "match")

        results = vfs.grep("match", path="/src", glob_filter="/src/*.py")

        assert len(results) == 1
        assert results[0]["path"] == "/src/a.py"


# --- Edit ---


class TestEdit:
    def test_single_replacement(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello world")

        count = vfs.edit("/f.txt", "hello", "hi")

        assert count == 1
        assert vfs.read("/f.txt") == "hi world"

    def test_replace_all(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "aaa")

        count = vfs.edit("/f.txt", "a", "b", replace_all=True)

        assert count == 3
        assert vfs.read("/f.txt") == "bbb"

    def test_multiple_without_replace_all_raises(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "aa")

        with pytest.raises(ValueError, match="2 occurrences"):
            vfs.edit("/f.txt", "a", "b")

    def test_not_found_raises(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello")

        with pytest.raises(ValueError, match="not found"):
            vfs.edit("/f.txt", "missing", "x")

    def test_file_not_found_raises(self):
        vfs = VirtualFilesystem()

        with pytest.raises(FileNotFoundError):
            vfs.edit("/nope.txt", "a", "b")

    def test_multiline_replacement(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "def foo():\n    pass")

        vfs.edit("/f.txt", "def foo():\n    pass", "def foo():\n    return 42")

        assert vfs.read("/f.txt") == "def foo():\n    return 42"

    def test_replace_with_empty_string(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "hello world")

        vfs.edit("/f.txt", " world", "")

        assert vfs.read("/f.txt") == "hello"

    def test_replace_with_longer_string(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "ab")

        vfs.edit("/f.txt", "ab", "abcdef")

        assert vfs.read("/f.txt") == "abcdef"

    def test_replace_all_returns_correct_count(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "the cat sat on the mat")

        count = vfs.edit("/f.txt", "the", "a", replace_all=True)

        assert count == 2

    def test_edit_normalizes_path(self):
        vfs = VirtualFilesystem()
        vfs.write("/a/b.txt", "old")

        vfs.edit("a/b.txt", "old", "new")

        assert vfs.read("/a/b.txt") == "new"


# --- Len and Files ---


class TestLen:
    def test_empty(self):
        assert len(VirtualFilesystem()) == 0

    def test_with_files(self):
        vfs = VirtualFilesystem()
        vfs.write("/a", "")
        vfs.write("/b", "")

        assert len(vfs) == 2

    def test_overwrite_does_not_increase_count(self):
        vfs = VirtualFilesystem()
        vfs.write("/a", "1")
        vfs.write("/a", "2")

        assert len(vfs) == 1

    def test_delete_decreases_count(self):
        vfs = VirtualFilesystem()
        vfs.write("/a", "")
        vfs.write("/b", "")
        vfs.delete("/a")

        assert len(vfs) == 1


class TestFilesProperty:
    def test_returns_copy(self):
        vfs = VirtualFilesystem()
        vfs.write("/a", "x")

        files = vfs.files
        files["/b"] = "y"

        assert not vfs.exists("/b")

    def test_contains_all_files(self):
        vfs = VirtualFilesystem()
        vfs.write("/a.txt", "a")
        vfs.write("/b.txt", "b")

        files = vfs.files

        assert len(files) == 2
        assert files["/a.txt"] == "a"
        assert files["/b.txt"] == "b"


# --- load_dict ---


class TestLoadDict:
    def test_loads_all_entries(self):
        vfs = VirtualFilesystem()
        vfs.load_dict({"/a.txt": "alpha", "/b.txt": "beta"})

        assert vfs.read("/a.txt") == "alpha"
        assert vfs.read("/b.txt") == "beta"

    def test_normalizes_paths(self):
        vfs = VirtualFilesystem()
        vfs.load_dict({"foo/bar.txt": "content"})

        assert vfs.exists("/foo/bar.txt")

    def test_empty_dict(self):
        vfs = VirtualFilesystem()
        vfs.load_dict({})

        assert len(vfs) == 0

    def test_overwrites_existing(self):
        vfs = VirtualFilesystem()
        vfs.write("/f.txt", "old")
        vfs.load_dict({"/f.txt": "new"})

        assert vfs.read("/f.txt") == "new"


# --- load_directory ---


class TestLoadDirectory:
    def test_loads_skill_files(self):
        vfs = VirtualFilesystem()
        count = vfs.load_directory(
            FIXTURES / "skills" / "market-sizing", base_path="/skill",
        )

        assert count >= 2
        assert vfs.exists("/skill/SKILL.md")
        assert vfs.exists("/skill/calculator.py")

    def test_content_matches_real_file(self):
        vfs = VirtualFilesystem()
        vfs.load_directory(
            FIXTURES / "skills" / "market-sizing", base_path="/s",
        )

        real = (FIXTURES / "skills" / "market-sizing" / "SKILL.md").read_text()
        assert vfs.read("/s/SKILL.md") == real

    def test_default_base_path_is_root(self):
        vfs = VirtualFilesystem()
        vfs.load_directory(FIXTURES / "skills" / "market-sizing")

        assert vfs.exists("/SKILL.md")

    def test_raises_on_non_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")

        vfs = VirtualFilesystem()
        with pytest.raises(NotADirectoryError):
            vfs.load_directory(f)

    def test_raises_on_nonexistent(self):
        vfs = VirtualFilesystem()
        with pytest.raises(NotADirectoryError):
            vfs.load_directory("/nonexistent/path")

    def test_recursive_subdirectories(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("top")
        (tmp_path / "sub" / "b.txt").write_text("nested")

        vfs = VirtualFilesystem()
        count = vfs.load_directory(tmp_path, base_path="/data")

        assert count == 2
        assert vfs.read("/data/a.txt") == "top"
        assert vfs.read("/data/sub/b.txt") == "nested"

    def test_returns_file_count(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        vfs = VirtualFilesystem()
        count = vfs.load_directory(tmp_path)

        assert count == 2
