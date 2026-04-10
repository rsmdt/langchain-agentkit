"""Edit tool — find-and-replace with quote normalization."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException

from langchain_agentkit.extensions.filesystem.tools.common import (
    _compute_structured_patch,
    _EditInput,
    _read_full_text,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# ---------------------------------------------------------------------------
# Quote normalization constants (ported from Claude Code utils.ts)
# ---------------------------------------------------------------------------

_LEFT_SINGLE_CURLY = "\u2018"  # '
_RIGHT_SINGLE_CURLY = "\u2019"  # '
_LEFT_DOUBLE_CURLY = "\u201c"  # "
_RIGHT_DOUBLE_CURLY = "\u201d"  # "

_CURLY_TO_STRAIGHT: dict[str, str] = {
    _LEFT_SINGLE_CURLY: "'",
    _RIGHT_SINGLE_CURLY: "'",
    _LEFT_DOUBLE_CURLY: '"',
    _RIGHT_DOUBLE_CURLY: '"',
}


# ---------------------------------------------------------------------------
# Quote normalization helpers
# ---------------------------------------------------------------------------


def _normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes for matching."""
    result = text
    for curly, straight in _CURLY_TO_STRAIGHT.items():
        result = result.replace(curly, straight)
    return result


def _is_opening_context(chars: list[str], index: int) -> bool:
    """Check if the character at index is in an opening quote context."""
    if index == 0:
        return True
    prev = chars[index - 1]
    return prev in (" ", "\t", "\n", "\r", "(", "[", "{", "\u2014", "\u2013")


def _apply_curly_double_quotes(text: str) -> str:
    """Replace straight double quotes with contextual curly double quotes."""
    chars = list(text)
    result: list[str] = []
    for i, ch in enumerate(chars):
        if ch == '"':
            result.append(
                _LEFT_DOUBLE_CURLY if _is_opening_context(chars, i) else _RIGHT_DOUBLE_CURLY,
            )
        else:
            result.append(ch)
    return "".join(result)


def _apply_curly_single_quotes(text: str) -> str:
    """Replace straight single quotes with contextual curly single quotes."""
    chars = list(text)
    result: list[str] = []
    for i, ch in enumerate(chars):
        if ch == "'":
            prev = chars[i - 1] if i > 0 else ""
            nxt = chars[i + 1] if i < len(chars) - 1 else ""
            prev_is_letter = bool(prev and unicodedata.category(prev).startswith("L"))
            next_is_letter = bool(nxt and unicodedata.category(nxt).startswith("L"))
            if prev_is_letter and next_is_letter:
                result.append(_RIGHT_SINGLE_CURLY)
            elif _is_opening_context(chars, i):
                result.append(_LEFT_SINGLE_CURLY)
            else:
                result.append(_RIGHT_SINGLE_CURLY)
        else:
            result.append(ch)
    return "".join(result)


def _preserve_quote_style(
    old_string: str,
    actual_old_string: str,
    new_string: str,
) -> str:
    """Apply the file's curly quote style to new_string when normalization was used."""
    if old_string == actual_old_string:
        return new_string

    has_double = _LEFT_DOUBLE_CURLY in actual_old_string or _RIGHT_DOUBLE_CURLY in actual_old_string
    has_single = _LEFT_SINGLE_CURLY in actual_old_string or _RIGHT_SINGLE_CURLY in actual_old_string

    if not has_double and not has_single:
        return new_string

    result = new_string
    if has_double:
        result = _apply_curly_double_quotes(result)
    if has_single:
        result = _apply_curly_single_quotes(result)
    return result


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """Find the actual string in file, falling back to quote-normalized matching."""
    if search_string in file_content:
        return search_string
    normalized_content = _normalize_quotes(file_content)
    normalized_search = _normalize_quotes(search_string)
    if normalized_search in normalized_content:
        pos = normalized_content.index(normalized_search)
        return file_content[pos : pos + len(normalized_search)]
    return None


# ---------------------------------------------------------------------------
# Trailing whitespace stripping
# ---------------------------------------------------------------------------


def _strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace per line, preserving line endings."""
    lines = re.split(r"(\r\n|\n|\r)", text)
    result: list[str] = []
    for i, part in enumerate(lines):
        if i % 2 == 0:
            result.append(part.rstrip())
        else:
            result.append(part)
    return "".join(result)


# ---------------------------------------------------------------------------
# Edit helpers
# ---------------------------------------------------------------------------


def _handle_empty_old_string(
    backend: Any,
    file_path: str,
    new_string: str,
) -> tuple[str, dict[str, Any]]:
    """Handle edit with empty old_string — file creation or filling empty file."""
    try:
        content = backend.read(file_path, limit=1)
        if content.strip():
            raise ToolException(
                f"File {file_path} is not empty. Cannot use empty old_string on a non-empty file."
            )
    except FileNotFoundError:
        pass  # File doesn't exist — will create
    backend.write(file_path, new_string)
    message = f"The file {file_path} has been updated successfully."
    artifact: dict[str, Any] = {
        "filePath": file_path,
        "oldString": "",
        "newString": new_string,
        "replaceAll": False,
        "originalFile": None,
    }
    return message, artifact


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_edit(backend: Any) -> BaseTool:  # noqa: C901
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Perform exact string replacement in a file."""
        if old_string == "":
            return _handle_empty_old_string(backend, file_path, new_string)

        # Read original file for quote normalization and artifact
        try:
            original_file = _read_full_text(backend, file_path)
        except FileNotFoundError as exc:
            raise ToolException(str(exc)) from exc

        # Strip trailing whitespace from new_string (except markdown)
        effective_new = new_string
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".md", ".mdx"):
            effective_new = _strip_trailing_whitespace(new_string)

        # Quote normalization: find actual string in file
        actual_old = _find_actual_string(original_file, old_string)
        if actual_old is None:
            raise ToolException(f"String not found in {file_path}.")

        # Preserve the file's quote style in new_string
        effective_new = _preserve_quote_style(old_string, actual_old, effective_new)

        # Trailing newline stripping on delete
        effective_old = actual_old
        if (
            effective_new == ""
            and not actual_old.endswith("\n")
            and actual_old + "\n" in original_file
        ):
            effective_old = actual_old + "\n"

        try:
            result = backend.edit(
                file_path,
                effective_old,
                effective_new,
                replace_all=replace_all,
            )
            count = result.get("replacements", 0) if isinstance(result, dict) else result
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        if count == 0:
            raise ToolException(f"String not found in {file_path}.")

        # Compute patch from in-memory replace (avoids re-reading the file)
        if replace_all:
            updated_file = original_file.replace(effective_old, effective_new)
        else:
            updated_file = original_file.replace(effective_old, effective_new, 1)
        patch = _compute_structured_patch(original_file, updated_file)

        artifact: dict[str, Any] = {
            "filePath": file_path,
            "oldString": old_string,
            "newString": new_string,
            "replaceAll": replace_all,
            "originalFile": original_file,
            "structuredPatch": patch,
        }

        if replace_all:
            message = (
                f"The file {file_path} has been updated. "
                f"All occurrences were successfully replaced."
            )
        else:
            message = f"The file {file_path} has been updated successfully."
        return message, artifact

    return StructuredTool.from_function(
        func=edit,
        name="Edit",
        description=(
            "Perform exact string replacements in a file. "
            "Fails if old_string is not unique (use replace_all for multiple)."
        ),
        args_schema=_EditInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
