"""Internal helpers for AgentKit's built-in reminder channel.

AgentKit always emits a ``<system-reminder>`` envelope as part of
:meth:`AgentKit.compose`. The only always-on built-in section is
today's date (``# currentDate``); extension-contributed sections
collected from ``Extension.prompt()`` dict returns with a
``"reminder"`` key are appended after it.

Users who want project-context files such as ``AGENTS.md`` injected
into the system prompt should pass them explicitly, e.g.
``AgentKit(prompt=Path("AGENTS.md"))`` or
``AgentKit(prompt=[Path("AGENTS.md"), Path("~/.agents/AGENTS.md").expanduser()])``.

This module exposes only private helpers; the reminder channel is not
configurable via a public dataclass.
"""

from __future__ import annotations

import datetime as _dt
import logging

_logger = logging.getLogger(__name__)


_REFERENCE_DISCLAIMER = (
    "IMPORTANT: this context may or may not be relevant to your tasks. "
    "You should not respond to this context unless it is highly relevant to your task."
)

_HEADER_LINE = "As you answer the user's questions, you can use the following context:"
_DATE_FORMAT = "Today's date is %Y-%m-%d."


def _format_date() -> str:
    try:
        return _dt.date.today().strftime(_DATE_FORMAT)
    except (ValueError, TypeError):
        return ""


def _wrap_envelope(sections: list[str]) -> str:
    """Render the ``<system-reminder>`` envelope around collected sections.

    Returns an empty string when no section yields content.
    """
    non_empty = [s for s in sections if s]
    if not non_empty:
        return ""
    body = "\n".join(non_empty)
    return f"<system-reminder>\n{_HEADER_LINE}\n{body}\n{_REFERENCE_DISCLAIMER}\n</system-reminder>"


def assemble_builtin_reminder(extra_sections: list[str] | None = None) -> str:
    """Assemble the built-in reminder payload.

    Always emits today's date. Extension-contributed reminder sections
    (already rendered as ``# key\\nvalue`` blocks) are appended after.
    """
    sections: list[str] = []
    today = _format_date()
    if today:
        sections.append(f"# currentDate\n{today}")
    if extra_sections:
        sections.extend(s for s in extra_sections if s)
    return _wrap_envelope(sections)
