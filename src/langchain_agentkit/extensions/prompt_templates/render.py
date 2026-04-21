"""Template expansion — bash-style positional substitution.

Supported placeholders (non-recursive — once substituted, values are
NOT re-expanded):

* ``$1`` .. ``$9`` — positional arguments, 1-indexed.
* ``${N}`` — same, for N >= 10 or disambiguation.
* ``$@`` / ``${@}`` — space-joined list of all positional args.
* ``${@:N:L}`` — slice starting at 1-indexed position ``N`` with length
  ``L``. ``N`` may be 0 (shifted to 1). Negative or out-of-range
  indices produce an empty slice.
* ``$*`` / ``${*}`` — alias for ``$@``.

Unresolved placeholders (e.g. ``$7`` with only 2 args) expand to empty
string rather than raising — templates should tolerate optional args.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Matches ``$N``, ``${N}``, ``$@``, ``${@}``, ``${@:N:L}``, ``$*``, ``${*}``.
_PATTERN = re.compile(
    r"\$(?:"
    r"\{@:(\d+):(\d+)\}"  # ${@:N:L}
    r"|\{(\d+)\}"  # ${N}
    r"|\{@\}"  # ${@}
    r"|\{\*\}"  # ${*}
    r"|(\d)"  # $N (single digit)
    r"|@"  # $@
    r"|\*"  # $*
    r")"
)


def expand_template(body: str, args: Sequence[str]) -> str:
    """Render ``body`` by substituting positional placeholders."""

    def _replace(match: re.Match[str]) -> str:
        slice_n, slice_l, braced_n, single_n = match.group(1, 2, 3, 4)
        text = match.group(0)
        if slice_n is not None and slice_l is not None:
            start = max(int(slice_n) - 1, 0)
            length = max(int(slice_l), 0)
            return " ".join(args[start : start + length])
        if braced_n is not None:
            idx = int(braced_n) - 1
            return args[idx] if 0 <= idx < len(args) else ""
        if single_n is not None:
            idx = int(single_n) - 1
            return args[idx] if 0 <= idx < len(args) else ""
        # $@ / $* / ${@} / ${*}
        if text in ("$@", "${@}", "$*", "${*}"):
            return " ".join(args)
        return text  # pragma: no cover — regex shouldn't match anything else

    return _PATTERN.sub(_replace, body)
