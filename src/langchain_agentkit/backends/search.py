from __future__ import annotations

import base64
import json
import re
from typing import TYPE_CHECKING, Any

from langchain_agentkit.backends.protocol import GrepMatch
from langchain_agentkit.backends.results import GlobResult, GrepResult
from langchain_agentkit.backends.shell import shell_quote

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "compile_glob_pattern",
    "glob_pattern_regex",
    "parse_remote_glob",
    "parse_remote_grep",
    "remote_glob_command",
    "remote_grep_command",
    "summarize_search_errors",
]

_REPORTED_ERROR_LIMIT = 5


def glob_pattern_regex(pattern: str) -> str:
    if "/" not in pattern:
        return "(?s)(?:.*/)?" + _segment_regex(pattern)
    segments = pattern.lstrip("/").split("/")
    last_index = len(segments) - 1
    parts: list[str] = []
    for index, segment in enumerate(segments):
        is_last = index == last_index
        if segment == "**":
            parts.append(".*" if is_last else "(?:.*/)?")
        else:
            parts.append(_segment_regex(segment) + ("" if is_last else "/"))
    return "(?s)" + "".join(parts)


def compile_glob_pattern(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(glob_pattern_regex(pattern))

    def matches(relative_path: str) -> bool:
        return compiled.fullmatch(relative_path) is not None

    return matches


def _segment_regex(segment: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            while index + 1 < len(segment) and segment[index + 1] == "*":
                index += 1
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[" and (close := segment.find("]", index + 2)) != -1:
            out.append(_bracket_regex(segment[index + 1 : close]))
            index = close
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def _bracket_regex(body: str) -> str:
    negated = body[:1] in ("!", "^")
    members = (body[1:] if negated else body).replace("\\", "\\\\").replace("[", "\\[")
    return f"[^/{members}]" if negated else f"[{members}]"


def summarize_search_errors(errors: list[str]) -> str | None:
    if not errors:
        return None
    shown = "; ".join(errors[:_REPORTED_ERROR_LIMIT])
    hidden = len(errors) - _REPORTED_ERROR_LIMIT
    return f"{shown}; and {hidden} more" if hidden > 0 else shown


_REMOTE_SEARCH_SCRIPT = """
import base64, json, os, re, sys
d = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
root = d['root']
limit = d['limit']
include = re.compile(d['include']) if d['include'] is not None else None
content = None
if d['content'] is not None:
    content = re.compile(d['content'], re.IGNORECASE if d['ignore_case'] else 0)
count = 0

def out(record):
    sys.stdout.write(json.dumps(record) + '\\n')

def finish(truncated):
    out({'done': True, 'truncated': truncated})
    sys.stdout.flush()
    sys.exit(0)

def failed(exc, path):
    out({'error': exc.strerror or str(exc), 'path': path})

def walk(directory):
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError as exc:
        failed(exc, directory)
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                yield from walk(entry.path)
            elif entry.is_file():
                yield entry.path
        except OSError as exc:
            failed(exc, entry.path)

if not os.path.isdir(root):
    out({'error': 'Directory not found', 'path': root})
    finish(False)
for path in walk(root):
    if include is not None and not include.fullmatch(os.path.relpath(path, root)):
        continue
    if content is None:
        if count == limit:
            finish(True)
        out({'path': path})
        count += 1
        continue
    try:
        with open(path, encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if content.search(line):
                    if count == limit:
                        finish(True)
                    out({'path': path, 'line': number, 'text': line})
                    count += 1
    except UnicodeDecodeError:
        continue
    except OSError as exc:
        failed(exc, path)
finish(False)
"""


def remote_glob_command(root: str, pattern: str, max_results: int) -> str:
    return _remote_search_command(
        {
            "root": root,
            "limit": max_results,
            "include": glob_pattern_regex(pattern),
            "content": None,
            "ignore_case": False,
        }
    )


def remote_grep_command(
    root: str,
    pattern: str,
    include_glob: str | None,
    ignore_case: bool,
    max_matches: int,
) -> str:
    return _remote_search_command(
        {
            "root": root,
            "limit": max_matches,
            "include": glob_pattern_regex(include_glob) if include_glob else None,
            "content": pattern,
            "ignore_case": ignore_case,
        }
    )


def parse_remote_glob(
    output: str,
    *,
    failure_detail: str,
    cut_short: bool,
    to_virtual: Callable[[str], str],
) -> GlobResult:
    records, truncated, errors = _parse_remote_records(
        output, failure_detail=failure_detail, cut_short=cut_short, to_virtual=to_virtual
    )
    return GlobResult(
        paths=[to_virtual(record["path"]) for record in records],
        truncated=truncated,
        error_message=summarize_search_errors(errors),
    )


def parse_remote_grep(
    output: str,
    *,
    failure_detail: str,
    cut_short: bool,
    to_virtual: Callable[[str], str],
) -> GrepResult:
    records, truncated, errors = _parse_remote_records(
        output, failure_detail=failure_detail, cut_short=cut_short, to_virtual=to_virtual
    )
    return GrepResult(
        matches=[
            GrepMatch(path=to_virtual(record["path"]), line=record["line"], text=record["text"])
            for record in records
        ],
        truncated=truncated,
        error_message=summarize_search_errors(errors),
    )


def _remote_search_command(request: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(request).encode("utf-8")).decode("ascii")
    return f"python3 -c {shell_quote(_REMOTE_SEARCH_SCRIPT)} {shell_quote(encoded)}"


def _parse_remote_records(
    output: str,
    *,
    failure_detail: str,
    cut_short: bool,
    to_virtual: Callable[[str], str],
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    finished = False
    truncated = False
    for line in output.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("done"):
            finished = True
            truncated = bool(record.get("truncated"))
        elif "error" in record:
            errors.append(f"{record['error']}: {to_virtual(record['path'])}")
        elif "path" in record:
            records.append(record)
    if not finished:
        if cut_short:
            truncated = True
        else:
            errors.append(f"Search did not complete: {failure_detail.strip() or 'no output'}")
    return records, truncated, errors
