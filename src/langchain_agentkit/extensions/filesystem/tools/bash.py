"""Bash tool for executing shell commands via a FilesystemProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.backends.protocol import SandboxProtocol


_BASH_DESCRIPTION = """
Run a shell command in the workspace. Use when you need to execute a shell command. Runs the command and returns its output.
"""


class _BashInput(BaseModel):
    command: str = Field(
        description="The command to execute.",
    )
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds.",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Clear, concise description of what this command does. "
            "Keep it brief (5-10 words) for simple commands."
        ),
    )


def _build_bash_tool(backend: SandboxProtocol) -> BaseTool:
    """Build the Bash tool for shell command execution."""

    async def _bash(
        command: str,
        timeout: int | None = None,
        description: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        result = await backend.execute(command, timeout=timeout)
        stdout = result.get("output", "")
        stderr = result.get("stderr", "")
        exit_code = result.get("exit_code", -1)
        truncated = result.get("truncated", False)
        output_path = result.get("output_path")
        lines_dropped = result.get("lines_dropped", 0)
        bytes_dropped = result.get("bytes_dropped", 0)

        if truncated:
            notice_parts = ["... (output truncated)"]
            if bytes_dropped:
                notice_parts.append(f"{bytes_dropped} bytes dropped from earlier output")
            if lines_dropped:
                notice_parts.append(f"{lines_dropped} leading lines dropped from the tail window")
            if output_path:
                notice_parts.append(
                    f"full transcript at {output_path} — use Read with offset/limit to paginate"
                )
            stdout += "\n" + "; ".join(notice_parts)

        artifact: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exitCode": exit_code,
            "interrupted": truncated,
            "outputPath": output_path,
            "linesDropped": lines_dropped,
            "bytesDropped": bytes_dropped,
        }

        if exit_code != 0:
            return f"Exit code {exit_code}\n{stdout}", artifact
        return stdout, artifact

    return StructuredTool.from_function(
        coroutine=_bash,
        name="Bash",
        description=_BASH_DESCRIPTION,
        args_schema=_BashInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
