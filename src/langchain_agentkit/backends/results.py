"""Result types for backend operations.

Backend methods return result dataclasses with optional ``error`` codes
instead of raising for expected, LLM-actionable failures. Truly
unexpected failures (network down, SDK bug) still raise.

The error vocabulary is part of the protocol. Tool-layer code branches
on stable ``Literal`` codes to produce code-aware messages for the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Error code vocabulary
# ---------------------------------------------------------------------------

FileError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
    "decode_error",
    "io_error",
]

EditError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
    "decode_error",
    "io_error",
    "old_string_not_found",
    "ambiguous_match",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReadResult:
    content: str | None = None
    error: FileError | None = None
    error_message: str | None = None


@dataclass(slots=True)
class ReadBytesResult:
    content: bytes | None = None
    error: FileError | None = None
    error_message: str | None = None


@dataclass(slots=True)
class WriteResult:
    path: str | None = None
    bytes_written: int | None = None
    error: FileError | None = None
    error_message: str | None = None


@dataclass(slots=True)
class EditResult:
    path: str | None = None
    replacements: int | None = None
    occurrences: int | None = None
    error: EditError | None = None
    error_message: str | None = None


@dataclass(slots=True)
class FileUploadResult:
    path: str
    bytes_written: int | None = None
    error: FileError | None = None
    error_message: str | None = None


@dataclass(slots=True)
class FileDownloadResult:
    path: str
    content: bytes | None = None
    error: FileError | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Sandbox environment
# ---------------------------------------------------------------------------

#: Tools probed on every backend's ``environment()``.
#:
#: Curated to the minimum set whose presence/absence meaningfully changes
#: LLM-generated commands. POSIX baseline (``grep``/``sed``/``awk`` etc.)
#: is assumed present — probing adds noise. Build/runtime tools
#: (``node``/``python3``/``make`` etc.) are task-derived: if the agent
#: needs them, it finds out from a failed ``execute`` call. Shell binaries
#: are redundant with the ``Shell:`` line in the ``<env>`` block.
PROBED_TOOLS: tuple[str, ...] = (
    "git",  # version control workflows skip when absent
    "rg",  # ripgrep present → preferred over grep
    "jq",  # JSON pipelines vs python3 -c fallback
)


@dataclass(slots=True, frozen=True)
class SandboxEnvironment:
    """Snapshot of the shell environment a ``SandboxBackend`` runs commands in.

    Surfaced to the LLM via the ``<env>`` block in the system prompt. The
    LLM uses this to pick correct shell flags (BSD vs GNU vs busybox),
    reach for the right tool (``rg`` over ``grep`` if available), and
    understand its working directory.

    Probed once per backend lifetime and cached. ``OSBackend`` populates
    instantly from ``platform``/``shutil``; remote backends issue a single
    shell probe.

    ``os`` carries ``uname -srm``-equivalent content: kernel name, kernel
    release, machine architecture as a single space-separated string.
    Examples: ``"Darwin 25.2.0 arm64"``, ``"Linux 6.6.4 x86_64"``,
    ``"Windows 10.0.22621 AMD64"``. Single field rather than separate
    platform/version/arch — the LLM reads text and the leading word
    encodes the platform family unambiguously.
    """

    os: str
    shell: str
    cwd: str
    available_tools: frozenset[str] = field(default_factory=frozenset)
