"""Shell-command construction helpers shared by command-routing backends.

``BubblewrapBackend``, ``DaytonaBackend``, and ``MirageBackend`` implement
file operations by routing shell commands through ``execute()``. The
quoting idiom, the bash startup-warning strip, and the environment probe
are identical across them and live here.
"""

from __future__ import annotations

import re

from langchain_agentkit.backends.results import PROBED_TOOLS, ReadResult

__all__ = [
    "BASH_STARTUP_WARNING_RE",
    "ENV_PROBE_SCRIPT",
    "ENV_PROBE_SEPARATOR",
    "parse_read_window",
    "read_window_command",
    "shell_quote",
]


def shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation (POSIX single-quote escape)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def read_window_command(path: str, offset: int, limit: int) -> str:
    quoted = shell_quote(path)
    count_lines = f"awk 'END {{ print NR }}' {quoted}"
    if limit <= 0:
        return count_lines
    start = max(0, offset) + 1
    end = start + limit - 1
    return f"{count_lines} && sed -n '{start},{end}p' {quoted}"


def parse_read_window(output: str) -> ReadResult:
    count, _, content = output.partition("\n")
    try:
        total_lines = int(count.strip())
    except ValueError:
        return ReadResult(error="io_error", error_message=f"Read failed: {output[:200]!r}")
    return ReadResult(content=content, total_lines=total_lines)


# Strip bash's ``setlocale: LC_ALL: cannot change locale`` startup warnings
# from stdout. On minimal Linux images the locale data isn't present, and
# bash emits the warning to stdout (not stderr) at startup *before* any
# in-shell ``export`` can adjust LC_ALL. Left unstripped it corrupts
# ``read``/``read_bytes`` content and the environment probe (whose first
# line is then the warning, not ``uname -srm``).
BASH_STARTUP_WARNING_RE = re.compile(
    r"^[^:\n]*: warning: setlocale:[^\n]*\n",
    re.MULTILINE,
)


# Single-shot environment probe. Output layout (line-oriented):
#   line 1:  uname -srm output (e.g. "Linux 5.15.0-91-generic x86_64")
#   line 2:  $SHELL value (or "/bin/sh" if unset)
#   line 3:  literal separator "TOOLS_BEGIN"
#   lines 4+: one available tool name per line, in PROBED_TOOLS order
#
# The separator is a plain identifier: some hosts run the probe under zsh,
# where a ``===``-style separator trips the ``=cmd`` path-expansion syntax.
ENV_PROBE_SEPARATOR = "TOOLS_BEGIN"
ENV_PROBE_SCRIPT = (
    "uname -srm\n"
    'echo "${SHELL:-/bin/sh}"\n'
    f"echo {ENV_PROBE_SEPARATOR}\n"
    "for t in " + " ".join(PROBED_TOOLS) + "; do "
    'command -v "$t" >/dev/null 2>&1 && echo "$t"; '
    "done\n"
    # Ensure the script always exits 0 so ``environment()`` parses the
    # well-formed output. Without this, the exit code is whatever the last
    # ``command -v`` returned — non-zero if the final probed tool is absent.
    "exit 0"
)
