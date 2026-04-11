"""Grep tool — regex search across files with multiple output modes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from langchain_agentkit.extensions.filesystem.tools.common import (
    _GrepInput,
    _read_full_text,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPE_GLOB_MAP: dict[str, str] = {
    "py": "*.py",
    "js": "*.js",
    "ts": "*.ts",
    "tsx": "*.tsx",
    "jsx": "*.jsx",
    "java": "*.java",
    "go": "*.go",
    "rs": "*.rs",
    "rust": "*.rs",
    "rb": "*.rb",
    "c": "*.c",
    "cpp": "*.cpp",
    "h": "*.h",
    "cs": "*.cs",
    "php": "*.php",
    "swift": "*.swift",
    "kt": "*.kt",
    "scala": "*.scala",
    "sh": "*.sh",
    "yaml": "*.yaml",
    "yml": "*.yml",
    "json": "*.json",
    "xml": "*.xml",
    "html": "*.html",
    "css": "*.css",
    "md": "*.md",
    "sql": "*.sql",
    "r": "*.r",
}


def _format_limit_info(shown: int, total: int, head_limit: int) -> str:
    """Format pagination info."""
    if shown >= total:
        return ""
    return f"\n(showing {shown} of {total} results, use offset to paginate)"


def _resolve_grep_glob(glob: str | None, file_type: str | None) -> str | None:
    """Resolve effective glob from explicit glob or type filter."""
    if glob is not None:
        return glob
    if file_type is not None:
        return _TYPE_GLOB_MAP.get(file_type, f"*.{file_type}")
    return None


def _resolve_grep_context(
    context: int | None,
    context_alias: int | None,
    before_context: int | None,
    after_context: int | None,
) -> tuple[int | None, int | None]:
    """Resolve context precedence: context > -C > -B/-A."""
    if context is not None:
        return context, context
    if context_alias is not None:
        return context_alias, context_alias
    return before_context, after_context


def _apply_offset_to_results(
    results: list[dict[str, Any]],
    offset: int,
    output_mode: str,
) -> list[dict[str, Any]]:
    """Skip first N entries, scoped by output mode."""
    if offset <= 0:
        return results
    if output_mode in ("files_with_matches", "count"):
        paths = list(dict.fromkeys(r["path"] for r in results))[offset:]
        path_set = set(paths)
        return [r for r in results if r["path"] in path_set]
    return results[offset:]


async def _grep_multiline(
    backend: Any,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool = False,
) -> list[dict[str, Any]]:
    """Full-file regex search with re.DOTALL for cross-line patterns."""
    flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
    regex = re.compile(pattern, flags)
    file_paths = await backend.glob(glob or "**/*", path=path or "/")
    results: list[dict[str, Any]] = []
    for file_path in file_paths:
        try:
            content = await _read_full_text(backend, file_path)
            for match in regex.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                matched_text = match.group()
                for i, line_text in enumerate(matched_text.split("\n")):
                    results.append(
                        {
                            "path": file_path,
                            "line": line_num + i,
                            "text": line_text,
                        }
                    )
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            continue
    return results


async def _format_grep_with_context(
    backend: Any,
    results: list[dict[str, Any]],
    before: int = 0,
    after: int = 0,
) -> list[str]:
    """Format grep results with surrounding context lines."""
    file_lines_cache: dict[str, list[str]] = {}
    output: list[str] = []

    for r in results:
        file_path = r["path"]
        line_num = r["line"]

        if file_path not in file_lines_cache:
            try:
                raw = await backend.read(file_path, limit=100_000)
                file_lines_cache[file_path] = raw.splitlines()
            except (FileNotFoundError, OSError):
                file_lines_cache[file_path] = []

        lines = file_lines_cache[file_path]
        start = max(0, line_num - 1 - before)
        end = min(len(lines), line_num + after)
        for i in range(start, end):
            prefix = ":" if i == line_num - 1 else "-"
            output.append(f"{file_path}{prefix}{i + 1}: {lines[i]}")
        output.append("--")

    return output


def _grep_files_with_matches(
    results: list[dict[str, Any]],
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    paths = list(dict.fromkeys(r["path"] for r in results))
    entries = paths[:head_limit] if head_limit else paths
    truncated = len(paths) > len(entries)
    content = f"Found {len(paths)} file(s)\n" + "\n".join(entries)
    if truncated:
        content += _format_limit_info(len(entries), len(paths), head_limit)
    artifact: dict[str, Any] = {
        "mode": "files_with_matches",
        "numFiles": len(paths),
        "filenames": entries,
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


def _grep_count(
    results: list[dict[str, Any]],
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["path"]] = counts.get(r["path"], 0) + 1
    items = list(counts.items())
    entries = items[:head_limit] if head_limit else items
    truncated = len(items) > len(entries)
    content = "\n".join(f"{p}: {c} match(es)" for p, c in entries)
    if truncated:
        content += _format_limit_info(len(entries), len(items), head_limit)
    total_matches = sum(c for _, c in entries)
    artifact: dict[str, Any] = {
        "mode": "count",
        "numFiles": len(items),
        "numMatches": total_matches,
        "filenames": [],
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


async def _format_content_results(
    backend: Any,
    results: list[dict[str, Any]],
    before: int | None,
    after: int | None,
    line_numbers: bool,
    head_limit: int,
    offset: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format grep results in content mode."""
    has_context = (before and before > 0) or (after and after > 0)
    if has_context:
        lines = await _format_grep_with_context(
            backend,
            results,
            before=before or 0,
            after=after or 0,
        )
    elif line_numbers:
        lines = [f"{r['path']}:{r['line']}: {r['text'].rstrip()}" for r in results]
    else:
        lines = [f"{r['path']}: {r['text'].rstrip()}" for r in results]
    total_lines = len(lines)
    truncated = head_limit > 0 and total_lines > head_limit
    if head_limit:
        lines = lines[:head_limit]
    content = "\n".join(lines)
    if truncated:
        content += _format_limit_info(len(lines), total_lines, head_limit)
    artifact: dict[str, Any] = {
        "mode": "content",
        "numFiles": len(set(r["path"] for r in results)),
        "numLines": len(lines),
        "filenames": [],
    }
    if truncated:
        artifact["appliedLimit"] = head_limit
    if offset > 0:
        artifact["appliedOffset"] = offset
    return content, artifact


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_grep(backend: Any) -> BaseTool:
    async def grep(  # noqa: PLR0913
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        context: int | None = None,
        before_context: int | None = None,
        after_context: int | None = None,
        context_alias: int | None = None,
        line_numbers: bool = True,
        ignore_case: bool = False,
        type: str | None = None,
        head_limit: int = 250,
        offset: int = 0,
        multiline: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Search file contents for a regex pattern."""
        effective_before, effective_after = _resolve_grep_context(
            context,
            context_alias,
            before_context,
            after_context,
        )
        effective_glob = _resolve_grep_glob(glob, type)

        if multiline:
            results = await _grep_multiline(
                backend,
                pattern,
                path=path,
                glob=effective_glob,
                ignore_case=ignore_case,
            )
        else:
            results = await backend.grep(
                pattern,
                path=path,
                glob=effective_glob,
                ignore_case=ignore_case,
            )
        if not results:
            return "No matches found.", {
                "mode": output_mode,
                "numFiles": 0,
                "filenames": [],
            }

        results = _apply_offset_to_results(results, offset, output_mode)

        if output_mode == "files_with_matches":
            return _grep_files_with_matches(results, head_limit, offset)
        if output_mode == "count":
            return _grep_count(results, head_limit, offset)
        return await _format_content_results(
            backend,
            results,
            effective_before,
            effective_after,
            line_numbers,
            head_limit,
            offset,
        )

    return StructuredTool.from_function(
        coroutine=grep,
        name="Grep",
        description=(
            "Search file contents for a regex pattern. "
            'Output modes: "content", "files_with_matches" (default), "count".'
        ),
        args_schema=_GrepInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
