"""Write tool — create or overwrite files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from langchain_agentkit.extensions.filesystem.tools.common import (
    _compute_structured_patch,
    _read_full_text,
    _WriteInput,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _build_write(backend: Any) -> BaseTool:
    async def write(file_path: str, content: str) -> tuple[str, dict[str, Any]]:
        """Write content to a file."""
        is_new = False
        original_file: str | None = None
        try:
            original_file = await _read_full_text(backend, file_path)
        except (FileNotFoundError, OSError):
            is_new = True

        await backend.write(file_path, content)
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
        description=(
            "Write content to a file, creating or overwriting it. Prefer Edit for modifications."
        ),
        args_schema=_WriteInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
