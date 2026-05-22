"""Write tool — create or overwrite files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException

from langchain_agentkit.extensions.filesystem.tools.common import (
    _FULL_READ_LIMIT,
    _compute_structured_patch,
    _WriteInput,
)

_WRITE_DESCRIPTION = """
Create a new file, or replace an existing file's contents entirely. Use when writing a file from scratch or doing a full rewrite. Replaces the whole file; it does not patch part of one.
"""

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.backends.protocol import FilesystemProtocol


def _build_write(backend: FilesystemProtocol) -> BaseTool:
    async def write(file_path: str, content: str) -> tuple[str, dict[str, Any]]:
        """Write content to a file."""
        read_result = await backend.read(file_path, limit=_FULL_READ_LIMIT)

        is_new = False
        original_file: str | None = None
        if read_result.error == "file_not_found":
            is_new = True
        elif read_result.error is not None:
            raise ToolException(
                f"Failed to read {file_path} before write: {read_result.error_message}"
            )
        else:
            original_file = read_result.content

        write_result = await backend.write(file_path, content)
        if write_result.error == "permission_denied":
            raise ToolException(f"Access denied writing {file_path}.")
        if write_result.error == "is_directory":
            raise ToolException(f"{file_path} is a directory, not a file.")
        if write_result.error is not None:
            raise ToolException(f"Failed to write {file_path}: {write_result.error_message}")

        op_type = "create" if is_new else "update"
        if is_new:
            message = f"File created successfully at: {file_path}"
        else:
            message = f"The file {file_path} has been updated successfully."

        patch = (
            _compute_structured_patch(original_file, content) if original_file is not None else []
        )
        artifact: dict[str, Any] = {
            "type": op_type,
            "filePath": file_path,
            "content": content,
            "structuredPatch": patch,
            "originalFile": original_file,
        }
        return message, artifact

    return StructuredTool.from_function(
        coroutine=write,
        name="Write",
        description=_WRITE_DESCRIPTION,
        args_schema=_WriteInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
