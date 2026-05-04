"""Backend helper functions — host-side conveniences over the protocol.

Free functions that *prepare* inputs for backend operations rather
than wrapping them. Keeping the actual ``backend.upload(...)`` call
visible at the call site preserves readability — the helper only
removes the boilerplate of walking a host directory.
"""

from __future__ import annotations

from pathlib import Path


def read_tree(
    local_path: str | Path,
    dest_path: str,
) -> list[tuple[str, bytes]]:
    """Read a host directory tree into ``backend.upload``-ready tuples.

    Walks ``local_path`` on the host, reads every regular file, and
    returns ``(dest, bytes)`` tuples preserving the relative tree
    rooted at ``dest_path``. Symlinks-to-files are followed
    (``Path.is_file`` semantics); empty directories are not included.

    Example::

        # Mirror examples/myapp/.agentkit/ into the backend at /.agentkit
        await backend.upload(
            read_tree(Path(__file__).parent / ".agentkit", "/.agentkit")
        )

    Args:
        local_path: Host directory to read from.
        dest_path: Virtual root path the tuples should target.

    Returns:
        ``list[tuple[str, bytes]]`` shaped for ``FileTransferBackend.upload``.
    """
    src = Path(local_path)
    dest = dest_path.rstrip("/")
    return [
        (f"{dest}/{f.relative_to(src).as_posix()}", f.read_bytes())
        for f in sorted(src.rglob("*"))
        if f.is_file()
    ]
