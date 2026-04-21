"""Tests for bounded execution capture (I5)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langchain_agentkit.backends._execution import BoundedCapture, _TailBuffer
from langchain_agentkit.backends.os import OSBackend


def test_tail_buffer_keeps_only_max_bytes() -> None:
    tb = _TailBuffer(max_bytes=100, max_lines=1000)
    tb.feed(b"x" * 500)
    result = tb.finalize()
    assert len(result.tail) == 100
    assert result.total_bytes == 500
    assert result.bytes_dropped == 400
    assert result.lines_dropped == 0


def test_tail_buffer_line_cap_trims_oldest() -> None:
    tb = _TailBuffer(max_bytes=10_000, max_lines=3)
    tb.feed(b"one\ntwo\nthree\nfour\nfive\n")
    result = tb.finalize()
    lines = result.tail.splitlines()
    # Oldest lines dropped; tail keeps last 3.
    assert lines == [b"three", b"four", b"five"]
    assert result.lines_dropped == 2


def test_tail_buffer_under_cap_returns_full_buffer() -> None:
    tb = _TailBuffer(max_bytes=1024, max_lines=1000)
    tb.feed(b"short output")
    result = tb.finalize()
    assert result.tail == b"short output"
    assert result.bytes_dropped == 0
    assert result.lines_dropped == 0


def test_bounded_capture_no_overflow_deletes_spill() -> None:
    cap = BoundedCapture(stdout_max_bytes=1024, stdout_max_lines=100)
    spill = cap.spill_path
    assert spill.exists()
    # Synchronous feed via the os.write path — drive the coroutine manually.
    import asyncio

    asyncio.run(cap.feed_stdout(b"hello\n"))
    asyncio.run(cap.feed_stderr(b"warn\n"))
    stdout_res, stderr_res, spill_path = cap.finalize()
    assert spill_path is None
    assert not spill.exists()
    assert stdout_res.tail == b"hello\n"
    assert stderr_res.tail == b"warn\n"


def test_bounded_capture_overflow_keeps_spill() -> None:
    import asyncio

    cap = BoundedCapture(stdout_max_bytes=32, stdout_max_lines=100)
    spill = cap.spill_path
    asyncio.run(cap.feed_stdout(b"A" * 100))  # overflows
    _stdout_res, _stderr_res, spill_path = cap.finalize()
    assert spill_path is not None
    assert spill.exists()
    assert spill.read_bytes() == b"A" * 100
    spill.unlink()  # cleanup


def test_bounded_capture_interleaves_streams_in_spill() -> None:
    import asyncio

    cap = BoundedCapture(stdout_max_bytes=4, stdout_max_lines=10)

    async def drive() -> None:
        await cap.feed_stdout(b"OUT1-")
        await cap.feed_stderr(b"ERR1-")
        await cap.feed_stdout(b"OUT2")

    asyncio.run(drive())
    _, _, spill_path = cap.finalize()
    assert spill_path is not None
    assert spill_path.read_bytes() == b"OUT1-ERR1-OUT2"
    spill_path.unlink()


def test_bounded_capture_abandon_cleans_up() -> None:
    cap = BoundedCapture()
    spill = cap.spill_path
    assert spill.exists()
    cap.abandon()
    assert not spill.exists()


@pytest.mark.asyncio
async def test_os_backend_execute_small_output(tmp_path: Path) -> None:
    backend = OSBackend(str(tmp_path))
    result = await backend.execute("echo hello")
    assert result["output"].strip() == "hello"
    assert result["exit_code"] == 0
    assert result["truncated"] is False
    assert result.get("output_path") is None
    assert result.get("lines_dropped") == 0
    assert result.get("bytes_dropped") == 0


@pytest.mark.asyncio
async def test_os_backend_execute_overflow_spills_to_file(tmp_path: Path) -> None:
    backend = OSBackend(str(tmp_path), max_output_bytes=128, max_output_lines=500)
    # Produce ~5 KiB of output on stdout via printf, well above 128 bytes.
    cmd = "python3 -c 'import sys; sys.stdout.write(\"z\" * 5000)'"
    result = await backend.execute(cmd)
    assert result["exit_code"] == 0
    assert result["truncated"] is True
    assert result.get("bytes_dropped", 0) > 0
    path = result.get("output_path")
    assert path is not None
    assert os.path.exists(path)
    content = Path(path).read_bytes()
    assert len(content) == 5000
    Path(path).unlink()


@pytest.mark.asyncio
async def test_os_backend_execute_exit_code(tmp_path: Path) -> None:
    backend = OSBackend(str(tmp_path))
    result = await backend.execute("exit 7")
    assert result["exit_code"] == 7


@pytest.mark.asyncio
async def test_os_backend_timeout_marks_truncated(tmp_path: Path) -> None:
    backend = OSBackend(str(tmp_path))
    result = await backend.execute("sleep 5", timeout=1)
    assert result["truncated"] is True
    assert result["exit_code"] == -1
