import asyncio
import base64
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agentkit.backends.results import (
    EditResult,
    GlobResult,
    GrepResult,
    ReadBytesResult,
    ReadResult,
    WriteResult,
)
from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools
from langchain_agentkit.permissions.types import (
    OperationPermissions,
    PermissionRule,
    PermissionRuleset,
)


class FakeBackend:
    def __init__(self, files: dict[str, str | bytes] | None = None) -> None:
        self.files: dict[str, str | bytes] = dict(files or {})
        self.glob_result = GlobResult()
        self.grep_result = GrepResult()
        self.total_lines_override: int | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.calls.append(("read", {"path": path, "offset": offset, "limit": limit}))
        if path not in self.files:
            return ReadResult(error="file_not_found")
        content = self.files[path]
        assert isinstance(content, str)
        lines = content.splitlines(keepends=True)
        total = self.total_lines_override if self.total_lines_override is not None else len(lines)
        return ReadResult(content="".join(lines[offset : offset + limit]), total_lines=total)

    async def read_bytes(self, path: str) -> ReadBytesResult:
        if path not in self.files:
            return ReadBytesResult(error="file_not_found")
        content = self.files[path]
        return ReadBytesResult(content=content if isinstance(content, bytes) else content.encode())

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        current = self.files.get(path)
        await asyncio.sleep(0)
        del current
        self.files[path] = content
        return WriteResult(path=path)

    async def edit(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        current = self.files[path]
        assert isinstance(current, str)
        await asyncio.sleep(0)
        if old_string not in current:
            return EditResult(error="old_string_not_found")
        self.files[path] = current.replace(old_string, new_string, -1 if replace_all else 1)
        return EditResult(path=path, replacements=1)

    async def glob(self, pattern: str, path: str = "/", max_results: int = 1000) -> GlobResult:
        self.calls.append(("glob", {"pattern": pattern, "path": path, "max_results": max_results}))
        return self.glob_result

    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
        max_matches: int = 1000,
    ) -> GrepResult:
        self.calls.append(("grep", {"pattern": pattern, "path": path, "max_matches": max_matches}))
        return self.grep_result


def _tool(backend: FakeBackend, name: str) -> Any:
    return next(t for t in create_filesystem_tools(backend) if t.name == name)


def _calls(backend: FakeBackend, kind: str) -> list[dict[str, Any]]:
    return [kwargs for k, kwargs in backend.calls if k == kind]


class TestGlobResultHandling:
    async def test_caps_backend_search_at_display_limit(self):
        backend = FakeBackend()
        await _tool(backend, "Glob").ainvoke({"pattern": "*.py"})

        assert _calls(backend, "glob")[0]["max_results"] == 100

    async def test_truncated_result_tells_model_results_are_partial(self):
        backend = FakeBackend()
        backend.glob_result = GlobResult(paths=["/a.py", "/b.py"], truncated=True)

        content, artifact = await _tool(backend, "Glob").coroutine(pattern="*.py")

        assert "/a.py" in content
        assert "truncated" in content.lower()
        assert artifact["truncated"] is True

    async def test_untruncated_result_has_no_partial_notice(self):
        backend = FakeBackend()
        backend.glob_result = GlobResult(paths=["/a.py"])

        content, artifact = await _tool(backend, "Glob").coroutine(pattern="*.py")

        assert content == "/a.py"
        assert artifact["truncated"] is False

    async def test_surfaces_error_message(self):
        backend = FakeBackend()
        backend.glob_result = GlobResult(paths=["/a.py"], error_message="/locked: unreadable")

        content, _ = await _tool(backend, "Glob").coroutine(pattern="*.py")

        assert "/locked: unreadable" in content

    async def test_surfaces_error_message_without_matches(self):
        backend = FakeBackend()
        backend.glob_result = GlobResult(error_message="/locked: unreadable")

        content, _ = await _tool(backend, "Glob").coroutine(pattern="*.py")

        assert "/locked: unreadable" in content


class TestGrepResultHandling:
    async def test_content_mode_passes_head_limit_plus_offset_as_max_matches(self):
        backend = FakeBackend()
        await _tool(backend, "Grep").ainvoke(
            {"pattern": "x", "output_mode": "content", "head_limit": 5, "offset": 3}
        )

        assert _calls(backend, "grep")[0]["max_matches"] == 8

    async def test_truncated_result_tells_model_results_are_partial(self):
        backend = FakeBackend()
        backend.grep_result = GrepResult(
            matches=[{"path": "/a.py", "line": 1, "text": "x"}], truncated=True
        )

        content, artifact = await _tool(backend, "Grep").coroutine(
            pattern="x", output_mode="content"
        )

        assert "/a.py:1: x" in content
        assert "partial" in content.lower()
        assert artifact["truncated"] is True

    async def test_truncated_files_mode_tells_model_results_are_partial(self):
        backend = FakeBackend()
        backend.grep_result = GrepResult(
            matches=[{"path": "/a.py", "line": 1, "text": "x"}], truncated=True
        )

        content, _ = await _tool(backend, "Grep").coroutine(pattern="x")

        assert "partial" in content.lower()

    async def test_surfaces_error_message(self):
        backend = FakeBackend()
        backend.grep_result = GrepResult(
            matches=[{"path": "/a.py", "line": 1, "text": "x"}],
            error_message="/locked: unreadable",
        )

        content, _ = await _tool(backend, "Grep").coroutine(pattern="x")

        assert "/locked: unreadable" in content

    async def test_surfaces_error_message_without_matches(self):
        backend = FakeBackend()
        backend.grep_result = GrepResult(error_message="invalid regex")

        content, _ = await _tool(backend, "Grep").coroutine(pattern="(")

        assert "invalid regex" in content

    async def test_multiline_reads_paths_from_glob_result(self):
        backend = FakeBackend({"/a.txt": "start\nhello\nworld\n"})
        backend.glob_result = GlobResult(paths=["/a.txt"])

        content, _ = await _tool(backend, "Grep").coroutine(
            pattern="hello.*world", output_mode="content", multiline=True
        )

        assert "/a.txt:2: hello" in content


class TestReadTotals:
    async def test_uses_backend_total_lines_and_names_next_offset(self):
        backend = FakeBackend({"/f.txt": "".join(f"line {i}\n" for i in range(30))})
        backend.total_lines_override = 500

        content, artifact = await _tool(backend, "Read").coroutine(
            file_path="/f.txt", offset=10, limit=10
        )

        assert artifact["totalLines"] == 500
        assert "offset=20" in content
        assert len(_calls(backend, "read")) == 1

    async def test_no_continuation_when_file_fully_shown(self):
        backend = FakeBackend({"/f.txt": "a\nb\n"})

        content, artifact = await _tool(backend, "Read").coroutine(file_path="/f.txt")

        assert artifact["totalLines"] == 2
        assert "offset=" not in content

    async def test_offset_past_end_reports_true_total(self):
        backend = FakeBackend({"/f.txt": "a\nb\n"})

        content, artifact = await _tool(backend, "Read").coroutine(file_path="/f.txt", offset=9)

        assert artifact["totalLines"] == 2
        assert "2 lines" in content

    async def test_negative_offset_clamped_to_zero(self):
        backend = FakeBackend({"/f.txt": "a\nb\n"})

        await _tool(backend, "Read").ainvoke({"file_path": "/f.txt", "offset": -5})

        assert _calls(backend, "read")[0]["offset"] == 0

    async def test_non_positive_limit_uses_default(self):
        backend = FakeBackend({"/f.txt": "a\nb\n"})

        result = await _tool(backend, "Read").ainvoke({"file_path": "/f.txt", "limit": 0})

        assert _calls(backend, "read")[0]["limit"] == 2000
        assert "a" in result


class TestReadGutter:
    async def test_two_space_gutter_keeps_source_tabs_unambiguous(self):
        backend = FakeBackend({"/f.py": "def f():\n\treturn 1\n"})

        result = await _tool(backend, "Read").ainvoke({"file_path": "/f.py"})

        assert result.splitlines() == ["1  def f():", "2  \treturn 1"]


class TestReadMultimodal:
    async def test_image_returned_as_image_content_block(self):
        data = b"\x89PNG\r\n\x1a\n"
        backend = FakeBackend({"/img.png": data})

        message = await _tool(backend, "Read").ainvoke(
            {"name": "Read", "args": {"file_path": "/img.png"}, "id": "c1", "type": "tool_call"}
        )

        assert isinstance(message, ToolMessage)
        [block] = message.content
        assert block["type"] == "image"
        assert block["base64"] == base64.b64encode(data).decode()
        assert block["mime_type"] == "image/png"

    async def test_pdf_returned_as_file_content_block(self):
        data = b"%PDF-1.4 fake"
        backend = FakeBackend({"/doc.pdf": data})

        message = await _tool(backend, "Read").ainvoke(
            {"name": "Read", "args": {"file_path": "/doc.pdf"}, "id": "c1", "type": "tool_call"}
        )

        blocks = [b for b in message.content if b["type"] == "file"]
        assert len(blocks) == 1
        assert blocks[0]["base64"] == base64.b64encode(data).decode()
        assert blocks[0]["mime_type"] == "application/pdf"
        assert message.artifact["filePath"] == "/doc.pdf"


def _image_tool_message() -> ToolMessage:
    return ToolMessage(
        content=[{"type": "image", "base64": "aGk=", "mime_type": "image/png"}],
        tool_call_id="c1",
        artifact={"filePath": "/img.png"},
        id="tm1",
    )


def _pdf_tool_message() -> ToolMessage:
    return ToolMessage(
        content=[{"type": "file", "base64": "aGk=", "mime_type": "application/pdf"}],
        tool_call_id="c2",
        artifact={"filePath": "/doc.pdf"},
        id="tm2",
    )


async def _ext_with_profile(profile: Any) -> FilesystemExtension:
    ext = FilesystemExtension(backend=FakeBackend())
    model = SimpleNamespace(profile=profile)
    await ext.setup(extensions=[ext], llm_getter=lambda: model)
    return ext


class TestUnsupportedContentFiltering:
    async def _run(self, ext: FilesystemExtension, messages: list[Any]) -> tuple[list[Any], Any]:
        seen: dict[str, Any] = {}

        async def handler(state: dict[str, Any]) -> dict[str, Any]:
            seen["messages"] = state["messages"]
            return {"messages": [*state["messages"], AIMessage(content="ok")]}

        result = await ext.wrap_model(state={"messages": messages}, handler=handler, runtime=None)
        return seen["messages"], result

    async def test_image_replaced_when_model_rejects_images(self):
        ext = await _ext_with_profile({"image_inputs": False})
        original = _image_tool_message()

        seen, _ = await self._run(ext, [original])

        [block] = seen[0].content
        assert block["type"] == "text"
        assert "/img.png" in block["text"]
        assert original.content[0]["type"] == "image"

    async def test_image_in_tool_message_replaced_when_tool_messages_reject_images(self):
        ext = await _ext_with_profile({"image_inputs": True, "image_tool_message": False})
        human = HumanMessage(
            content=[{"type": "image", "base64": "aGk=", "mime_type": "image/png"}]
        )

        seen, _ = await self._run(ext, [human, _image_tool_message()])

        assert seen[0] is human
        assert seen[1].content[0]["type"] == "text"

    async def test_pdf_replaced_when_model_rejects_pdfs(self):
        ext = await _ext_with_profile({"pdf_inputs": False})

        seen, _ = await self._run(ext, [_pdf_tool_message()])

        assert seen[0].content[0]["type"] == "text"
        assert "/doc.pdf" in seen[0].content[0]["text"]

    async def test_missing_profile_counts_as_supported(self):
        ext = await _ext_with_profile(None)
        original = _image_tool_message()

        seen, _ = await self._run(ext, [original])

        assert seen[0] is original

    async def test_missing_field_counts_as_supported(self):
        ext = await _ext_with_profile({"pdf_inputs": False})
        original = _image_tool_message()

        seen, _ = await self._run(ext, [original])

        assert seen[0] is original

    async def test_filtered_copies_never_reach_returned_state(self):
        ext = await _ext_with_profile({"image_inputs": False})
        original = _image_tool_message()

        _, result = await self._run(ext, [original])

        assert result["messages"][0] is original

    async def test_no_model_leaves_messages_untouched(self):
        ext = FilesystemExtension(backend=FakeBackend())
        original = _image_tool_message()

        seen, _ = await self._run(ext, [original])

        assert seen[0] is original


class TestConcurrentWritesSamePath:
    async def test_parallel_edits_to_one_file_are_both_applied(self):
        backend = FakeBackend({"/f.txt": "alpha\nbeta\n"})
        edit = _tool(backend, "Edit")

        await asyncio.gather(
            edit.ainvoke({"file_path": "/f.txt", "old_string": "alpha", "new_string": "ALPHA"}),
            edit.ainvoke({"file_path": "/f.txt", "old_string": "beta", "new_string": "BETA"}),
        )

        assert backend.files["/f.txt"] == "ALPHA\nBETA\n"

    async def test_edits_through_extension_tools_are_serialized(self):
        backend = FakeBackend({"/f.txt": "alpha\nbeta\n"})
        ext = FilesystemExtension(backend=backend)
        edit = next(t for t in ext.tools if t.name == "Edit")

        await asyncio.gather(
            edit.ainvoke({"file_path": "/f.txt", "old_string": "alpha", "new_string": "ALPHA"}),
            edit.ainvoke({"file_path": "/f.txt", "old_string": "beta", "new_string": "BETA"}),
        )

        assert backend.files["/f.txt"] == "ALPHA\nBETA\n"


def _ext_denying(operation: str, pattern: str) -> FilesystemExtension:
    rules = OperationPermissions(
        default="allow", rules=[PermissionRule(pattern=pattern, action="deny")]
    )
    ruleset = PermissionRuleset(default="allow", **{operation: rules})
    return FilesystemExtension(backend=FakeBackend(), permissions=ruleset)


def _ext_tool(ext: FilesystemExtension, name: str) -> Any:
    return next(t for t in ext.tools if t.name == name)


class TestSearchPermissionTargets:
    async def test_grep_regex_resembling_denied_path_is_not_blocked(self):
        ext = _ext_denying("grep", "/secret/**")

        result = await _ext_tool(ext, "Grep").ainvoke({"pattern": "/secret/key", "path": "/pub"})

        assert "Permission denied" not in result

    async def test_grep_on_denied_path_is_blocked(self):
        ext = _ext_denying("grep", "/secret/**")

        result = await _ext_tool(ext, "Grep").ainvoke({"pattern": "key", "path": "/secret/dir"})

        assert "Permission denied" in result

    async def test_grep_without_path_is_checked_against_root(self):
        ext = _ext_denying("grep", "/")

        result = await _ext_tool(ext, "Grep").ainvoke({"pattern": "key"})

        assert "Permission denied" in result

    async def test_glob_pattern_resembling_denied_path_is_not_blocked(self):
        ext = _ext_denying("glob", "/secret/**")

        result = await _ext_tool(ext, "Glob").ainvoke({"pattern": "/secret/**", "path": "/pub"})

        assert "Permission denied" not in result

    async def test_glob_without_path_is_checked_against_root_and_hints_path(self):
        ext = _ext_denying("glob", "/")

        result = await _ext_tool(ext, "Glob").ainvoke({"pattern": "src/*.py"})

        assert "Permission denied" in result
        assert "path" in result.split("Permission denied", 1)[1]

    async def test_glob_on_denied_path_is_blocked(self):
        ext = _ext_denying("glob", "/secret/**")

        result = await _ext_tool(ext, "Glob").ainvoke({"pattern": "*.py", "path": "/secret/x"})

        assert "Permission denied" in result
