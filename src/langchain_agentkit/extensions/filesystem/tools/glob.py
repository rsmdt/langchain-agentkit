"""Glob tool — fast file pattern matching."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from langchain_agentkit.extensions.filesystem.tools.common import (
    _GLOB_DEFAULT_LIMIT,
    _GlobInput,
)

_GLOB_DESCRIPTION = """
Find files by name or path pattern. Use when you know a filename or shape but not where it lives. A pattern without a "/" matches file names at any depth. Returns matching paths only; it does not read file contents or search inside them.
"""

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _build_glob(backend: Any) -> BaseTool:
    async def glob(pattern: str, path: str | None = None) -> tuple[str, dict[str, Any]]:
        """Find files matching a glob pattern."""
        start = time.monotonic()
        result = await backend.glob(pattern, path=path or "/", max_results=_GLOB_DEFAULT_LIMIT)
        duration_ms = round((time.monotonic() - start) * 1000)

        matches = result.paths[:_GLOB_DEFAULT_LIMIT]
        truncated = result.truncated or len(result.paths) > _GLOB_DEFAULT_LIMIT
        lines = list(matches) if matches else ["No files found"]
        if truncated:
            lines.append(
                "(Results are truncated and partial. Narrow the search with a more "
                "specific path or pattern.)"
            )
        if result.error_message:
            lines.append(f"(Some locations could not be searched: {result.error_message})")
        artifact: dict[str, Any] = {
            "numFiles": len(matches),
            "filenames": matches,
            "truncated": truncated,
            "durationMs": duration_ms,
        }
        return "\n".join(lines), artifact

    return StructuredTool.from_function(
        coroutine=glob,
        name="Glob",
        description=_GLOB_DESCRIPTION,
        args_schema=_GlobInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
