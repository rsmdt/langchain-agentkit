"""Write tool — create or overwrite files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException

from langchain_agentkit.extensions.filesystem.tools.common import (
    _FULL_READ_LIMIT,
    _compute_structured_patch,
    _WriteInput,
)

_WRITE_DESCRIPTION = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.backends.protocol import BackendProtocol


def _build_write(backend: BackendProtocol) -> BaseTool:
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
