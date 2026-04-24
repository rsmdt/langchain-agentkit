"""Shared execution helpers — bounded capture with temp-file spillover.

Bash-style shell output is unbounded: a single ``pytest -v`` or
``npm install`` can emit megabytes. The naive ``proc.communicate()``
approach buffers everything into memory and then concatenates it into
the tool result, which blows the context window (or the process RSS).

:class:`BoundedCapture` streams stdout/stderr into a bounded in-memory
tail window while mirroring everything to a temp file on disk. The tool
layer surfaces the tail to the model and surfaces the temp path so the
model can ``Read`` it with offset/limit when needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["BoundedCapture", "CaptureResult", "drain_stream_into", "iter_lines"]

# Default tail window — 256 KiB or 2000 lines, whichever comes first.
DEFAULT_MAX_OUTPUT_BYTES: int = 256 * 1024
DEFAULT_MAX_OUTPUT_LINES: int = 2000

# Directory marker so orphan temp files are easy to identify / reap.
_TEMP_PREFIX = "agentkit-exec-"
_TEMP_SUFFIX = ".log"


@dataclass(slots=True)
class CaptureResult:
    """Final state of a :class:`BoundedCapture` after the stream closes."""

    tail: bytes
    total_bytes: int
    total_lines: int
    lines_dropped: int
    bytes_dropped: int


class _TailBuffer:
    """Rolling in-memory tail capped by byte length.

    Internally keeps up to ``2 * max_bytes`` so that line-aware trimming
    at finalize time still has enough context to snap to a line boundary.
    """

    __slots__ = ("_buf", "_lines_cap", "_spillway", "_tail_max", "_total_bytes", "_total_lines")

    def __init__(self, max_bytes: int, max_lines: int) -> None:
        self._tail_max = max_bytes
        self._lines_cap = max_lines
        self._spillway = max_bytes * 2
        self._buf = bytearray()
        self._total_bytes = 0
        self._total_lines = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._total_bytes += len(chunk)
        self._total_lines += chunk.count(b"\n")
        self._buf.extend(chunk)
        if len(self._buf) > self._spillway:
            del self._buf[: len(self._buf) - self._spillway]

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def finalize(self) -> CaptureResult:
        tail = (
            bytes(self._buf[-self._tail_max :])
            if len(self._buf) > self._tail_max
            else bytes(self._buf)
        )
        lines = tail.splitlines(keepends=True)
        lines_dropped = 0
        if len(lines) > self._lines_cap:
            dropped = len(lines) - self._lines_cap
            lines_dropped = dropped
            tail = b"".join(lines[-self._lines_cap :])
        bytes_dropped = self._total_bytes - len(tail)
        return CaptureResult(
            tail=tail,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            lines_dropped=lines_dropped,
            bytes_dropped=bytes_dropped,
        )


class BoundedCapture:
    """Coordinate bounded tails for stdout/stderr plus a combined spillover file.

    Always opens a temp file up front so the full interleaved transcript
    is preserved for the lifetime of the call. The file is deleted at
    :meth:`finalize` when nothing overflowed so the happy path leaves no
    litter. On overflow the caller surfaces :attr:`spill_path` to the
    tool layer.
    """

    def __init__(
        self,
        *,
        stdout_max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        stdout_max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
        stderr_max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        stderr_max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    ) -> None:
        self._stdout = _TailBuffer(stdout_max_bytes, stdout_max_lines)
        self._stderr = _TailBuffer(stderr_max_bytes, stderr_max_lines)
        fd, path = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX)
        self._spill_fd: int | None = fd
        self._spill_path: Path = Path(path)
        self._lock = asyncio.Lock()
        self._finalized = False

    @property
    def spill_path(self) -> Path:
        return self._spill_path

    async def feed_stdout(self, chunk: bytes) -> None:
        await self._feed(chunk, self._stdout)

    async def feed_stderr(self, chunk: bytes) -> None:
        await self._feed(chunk, self._stderr)

    async def _feed(self, chunk: bytes, tail: _TailBuffer) -> None:
        if not chunk:
            return
        tail.feed(chunk)
        # Mirror raw bytes to the spill file under a lock so stdout/stderr
        # interleave in write order rather than racing.
        async with self._lock:
            if self._spill_fd is not None:
                os.write(self._spill_fd, chunk)

    def finalize(self) -> tuple[CaptureResult, CaptureResult, Path | None]:
        """Close the spill file and return per-stream results + spill path.

        ``spill_path`` is ``None`` when neither stream overflowed — the
        temp file is unlinked in that case. Otherwise it points to the
        full interleaved transcript and ownership transfers to the caller.
        """
        if self._finalized:
            raise RuntimeError("BoundedCapture.finalize() called twice")
        self._finalized = True
        if self._spill_fd is not None:
            os.close(self._spill_fd)
            self._spill_fd = None
        stdout_result = self._stdout.finalize()
        stderr_result = self._stderr.finalize()
        overflowed = stdout_result.bytes_dropped > 0 or stderr_result.bytes_dropped > 0
        if not overflowed:
            with contextlib.suppress(OSError):
                self._spill_path.unlink()
            return stdout_result, stderr_result, None
        return stdout_result, stderr_result, self._spill_path

    def abandon(self) -> None:
        """Close and delete the spill file without finalizing results.

        Call when the caller raises before :meth:`finalize` — keeps the
        happy path clean and avoids temp-file leaks on cancellation.
        """
        if self._finalized:
            return
        self._finalized = True
        if self._spill_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._spill_fd)
            self._spill_fd = None
        with contextlib.suppress(OSError):
            self._spill_path.unlink()


async def drain_stream_into(
    stream: asyncio.StreamReader | None,
    consumer: Consumer,
    *,
    chunk_size: int = 8192,
) -> None:
    """Read ``stream`` in chunks and forward them to ``consumer``.

    Runs until EOF. Tolerates ``None`` (some platforms omit a stream
    when the child process doesn't open it).
    """
    if stream is None:
        return
    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            return
        await consumer(chunk)


# Minimal Consumer protocol for drain_stream_into; kept as a type alias so
# callers can pass either ``capture.feed_stdout`` / ``.feed_stderr`` directly.
if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    Consumer = Callable[[bytes], Awaitable[None]]


def iter_lines(tail: bytes) -> Iterable[str]:
    """Decode and yield lines from a tail buffer, UTF-8 with replacement."""
    text = tail.decode("utf-8", errors="replace")
    yield from text.splitlines()
