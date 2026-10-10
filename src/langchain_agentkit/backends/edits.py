from __future__ import annotations

from typing import Any

from langchain_agentkit.backends.results import EditResult

__all__ = ["edit_result_from_script", "empty_old_string_result"]


def empty_old_string_result(path: str) -> EditResult:
    return EditResult(
        path=path,
        error="empty_old_string",
        error_message="old_string must not be empty. Provide the exact text to replace.",
    )


def edit_result_from_script(data: dict[str, Any], path: str) -> EditResult:
    if "error" in data:
        err = data["error"]
        if err == "file_not_found":
            return EditResult(error="file_not_found", error_message=f"File not found: {path}")
        if err == "is_directory":
            return EditResult(error="is_directory", error_message=f"Path is a directory: {path}")
        if err == "decode_error":
            return EditResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path}",
            )
        if err == "old_string_not_found":
            return EditResult(
                path=path,
                error="old_string_not_found",
                error_message=f"old_string not found in {path}",
            )
        if err == "ambiguous_match":
            occ = data.get("occurrences")
            return EditResult(
                path=path,
                occurrences=occ,
                error="ambiguous_match",
                error_message=(
                    f"old_string appears {occ} times in {path}. "
                    "Pass replace_all=True or extend old_string for uniqueness."
                ),
            )
        return EditResult(
            error="io_error",
            error_message=data.get("message") or f"Edit failed: {err}",
        )
    return EditResult(path=path, replacements=data["replacements"])
