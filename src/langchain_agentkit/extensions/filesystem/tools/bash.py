"""Bash tool for executing shell commands via a BackendProtocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.backends.protocol import BackendProtocol


_BASH_DESCRIPTION = """Executes a given bash command and returns its output.

The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).

IMPORTANT: Avoid using this tool to run `find`, `grep`, `rg`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool as this will provide a much better experience for the user:

 - File search: Use Glob (NOT find or ls)
 - Content search: Use Grep (NOT grep or rg)
 - Read files: Use Read (NOT cat/head/tail)
 - Edit files: Use Edit (NOT sed/awk)
 - Write files: Use Write (NOT echo >/cat <<EOF)
 - Communication: Output text directly (NOT echo/printf)

While the Bash tool can do similar things, it's better to use the built-in tools as they provide a better user experience and make it easier to review tool calls and give permission.

# Instructions
 - If your command will create new directories or files, first verify the parent directory exists and is the correct location.
 - Always quote file paths that contain spaces with double quotes (e.g., "path with spaces/file.txt").
 - Try to maintain your current working directory throughout the session by using absolute paths and avoiding unnecessary `cd`.
 - You may specify an optional timeout (in seconds). If omitted, a reasonable default is used.
 - For long-running commands, consider running in the background when supported and polling for completion rather than blocking.
 - When issuing multiple commands:
   - If the commands are independent and can run in parallel, issue them as separate tool calls in a single turn.
   - If the commands depend on each other and must run sequentially, chain them with `&&` in one call.
   - Use `;` only when later commands should run regardless of earlier failures.
   - Do NOT use newlines to separate commands (newlines are ok inside quoted strings).
 - Avoid unnecessary `sleep` commands. Do not poll with short sleeps in a tight loop — prefer the longest interval that still meets the requirement, or wait on an explicit completion signal."""


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


def _build_bash_tool(backend: BackendProtocol) -> BaseTool:
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
        if truncated:
            stdout += "\n... (output truncated)"

        artifact: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exitCode": exit_code,
            "interrupted": truncated,
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
