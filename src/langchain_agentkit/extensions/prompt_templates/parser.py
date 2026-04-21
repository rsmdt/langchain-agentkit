"""Argument parser — bash-style quoting for template invocations.

Supports double- and single-quoted strings (no nesting), backslash
escapes inside double quotes, and whitespace-separated positional tokens.
This is intentionally simpler than POSIX shell word splitting — templates
don't need pipes, redirections, or variable expansion at parse time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class _ArgError(ValueError):
    """Raised on malformed quoted input."""


@dataclass(frozen=True, slots=True)
class ParsedArgs:
    """Result of :func:`parse_args`.

    ``positional`` keeps the order and count of tokens so ``$1``, ``$2``,
    ``$@`` and ``${@:N:L}`` can all be resolved from it.
    """

    positional: tuple[str, ...]
    raw: str

    @property
    def count(self) -> int:
        return len(self.positional)


def parse_args(raw: Any) -> ParsedArgs:
    """Tokenize ``raw`` into a positional argument list.

    Accepts a string to parse or a sequence of strings to use verbatim
    (handy when the caller already has structured args, e.g. from a
    programmatic ``kit.templates.render(name, ["x", "y"])`` call).
    """
    if raw is None or raw == "":
        return ParsedArgs(positional=(), raw="")
    if isinstance(raw, (list, tuple)):
        return ParsedArgs(positional=tuple(str(x) for x in raw), raw=" ".join(str(x) for x in raw))
    if not isinstance(raw, str):
        raise _ArgError(f"Unsupported args type: {type(raw).__name__}")

    tokens: list[str] = []
    i = 0
    n = len(raw)
    buf: list[str] = []
    in_token = False
    in_single = False
    in_double = False

    while i < n:
        ch = raw[i]
        if in_single:
            if ch == "'":
                in_single = False
            else:
                buf.append(ch)
            i += 1
            continue
        if in_double:
            if ch == "\\" and i + 1 < n:
                nxt = raw[i + 1]
                if nxt in ('"', "\\", "$", "`", "\n"):
                    buf.append(nxt)
                    i += 2
                    continue
                buf.append(ch)
                i += 1
                continue
            if ch == '"':
                in_double = False
            else:
                buf.append(ch)
            i += 1
            continue
        # Unquoted context
        if ch.isspace():
            if in_token:
                tokens.append("".join(buf))
                buf.clear()
                in_token = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            in_token = True
        elif ch == '"':
            in_double = True
            in_token = True
        elif ch == "\\" and i + 1 < n:
            buf.append(raw[i + 1])
            in_token = True
            i += 2
            continue
        else:
            buf.append(ch)
            in_token = True
        i += 1

    if in_single or in_double:
        raise _ArgError("Unterminated quoted string in template args")
    if in_token:
        tokens.append("".join(buf))
    return ParsedArgs(positional=tuple(tokens), raw=raw)
