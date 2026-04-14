"""Glob tool — fast file pattern matching."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from langchain_agentkit.extensions.filesystem.tools.common import (
    _GLOB_DEFAULT_LIMIT,
    _GlobInput,
)

_GLOB_DESCRIPTION = """- Fast file pattern matching tool that works with any codebase size
- Supports glob patterns like "**/*.js" or "src/**/*.ts"
- Returns matching file paths sorted by modification time
- Use this tool when you need to find files by name patterns
- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead"""

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _build_glob(backend: Any) -> BaseTool:
    async def glob(pattern: str, path: str | None = None) -> tuple[str, dict[str, Any]]:
        """Find files matching a glob pattern."""
        start = time.monotonic()
        all_matches = await backend.glob(pattern, path=path or "/")
        duration_ms = round((time.monotonic() - start) * 1000)

        if not all_matches:
            return "No files found", {
                "numFiles": 0,
                "filenames": [],
                "truncated": False,
                "durationMs": duration_ms,
            }

        truncated = len(all_matches) > _GLOB_DEFAULT_LIMIT
        matches = all_matches[:_GLOB_DEFAULT_LIMIT]
        content = "\n".join(matches)
        if truncated:
            content += "\n(Results are truncated. Consider using a more specific path or pattern.)"
        artifact: dict[str, Any] = {
            "numFiles": len(all_matches),
            "filenames": matches,
            "truncated": truncated,
            "durationMs": duration_ms,
        }
        return content, artifact

    return StructuredTool.from_function(
        coroutine=glob,
        name="Glob",
        description=_GLOB_DESCRIPTION,
        args_schema=_GlobInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
