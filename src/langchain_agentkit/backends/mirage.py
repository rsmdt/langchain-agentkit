"""MirageBackend — Unified VFS backend powered by Mirage Workspace.

Implements ``FilesystemProtocol`` and ``SandboxProtocol`` over a
pre-constructed :class:`mirage.Workspace`. File operations route
through the public ``ws.ops`` property; shell execution goes through
``ws.execute()`` (tree-sitter-bash parsed, in-process). Mirage's
killer feature — pipelines that span multiple mounts (S3, Slack,
GitHub, GDrive, …) — is preserved by claiming ``SandboxProtocol`` so
the framework registers the Bash tool.

Requires ``pip install langchain-agentkit[mirage]`` (or
``pip install mirage-ai`` directly). Importing this module fails with
a clear ``ImportError`` if the SDK isn't installed — ``MirageBackend``
isn't re-exported from the generic ``langchain_agentkit.backends``
namespace, so SDK-less users running ``OSBackend`` are unaffected.

The caller owns the workspace lifecycle: mount resources with the
appropriate :class:`mirage.MountMode` (``READ`` is default — pass
``MountMode.WRITE`` for any mount the agent should be able to write
to), pass the ``Workspace`` in fully constructed, and
``await ws.close()`` when done. This backend is a thin protocol
adapter, not a workspace manager.

Shell-execution caveat: Mirage's ``execute()`` is bash-grammar parsed
but runs a curated subset of POSIX tools in-process. ``available_tools``
in the :class:`SandboxEnvironment` reports the actual command vocabulary
exposed to the LLM (typically dozens of names plus any resource-specific
commands). Builtins like ``exit`` and external binaries (``make``,
``python``, ``git``) are not available — the shell is signaled via
``shell="mirage-bash"`` so a well-prompted LLM can avoid them.

Usage::

    from mirage import MountMode, RAMResource, Workspace
    from langchain_agentkit.backends.mirage import MirageBackend

    ws = Workspace({"/": (RAMResource(), MountMode.WRITE)})
    backend = MirageBackend(ws)

    result = await backend.read("/notes.md")
    if result.error is None:
        print(result.content)

    await ws.close()
"""

from __future__ import annotations

import asyncio
import contextlib
import platform
import posixpath
from typing import TYPE_CHECKING

# Runtime import — fails with a clear ImportError if ``mirage-ai`` isn't
# installed. Matches the convention in :mod:`agentfs` and :mod:`daytona`,
# where SDK absence is signaled at module import time rather than at
# first use.
from mirage import Workspace  # type: ignore[import-untyped] # noqa: TC002

from langchain_agentkit.backends.execution import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    BoundedCapture,
)
from langchain_agentkit.backends.protocol import ExecuteResponse, GrepMatch
from langchain_agentkit.backends.results import (
    EditResult,
    FileDownloadResult,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    WriteResult,
)

if TYPE_CHECKING:
    from langchain_agentkit.backends.results import FileError


def _shell_quote(s: str) -> str:
    """Quote a string for safe POSIX-shell interpolation."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# Map Python exception types to the framework's stable FileError
# vocabulary. Order matters: more specific subclasses must precede more
# general ones so the first matching isinstance() check wins.
# IsADirectoryError, NotADirectoryError, FileNotFoundError, and
# PermissionError are all subclasses of OSError; placing OSError last
# (via the fallback in :func:`_map_exc`) ensures it catches the residue.
_EXC_TO_FILE_ERROR: list[tuple[type[Exception], FileError]] = [
    (FileNotFoundError, "file_not_found"),
    (IsADirectoryError, "is_directory"),
    (NotADirectoryError, "invalid_path"),
    (PermissionError, "permission_denied"),
    (UnicodeDecodeError, "decode_error"),
]


def _map_exc(exc: Exception) -> FileError:
    for cls, code in _EXC_TO_FILE_ERROR:
        if isinstance(exc, cls):
            return code
    return "io_error"


class MirageBackend:
    """Mirage Workspace backend.

    Wraps a pre-constructed :class:`mirage.Workspace`. The caller is
    responsible for mounting resources and calling ``await ws.close()``
    when finished.

    Args:
        workspace: Pre-constructed Mirage ``Workspace`` with whichever
            resources mounted under whichever prefixes the application
            wants the agent to see.
        workdir: Absolute path used as the virtual root for path
            resolution. Defaults to ``"/"`` so the agent sees the whole
            multi-mount workspace. Pass a sub-path (e.g. ``"/workspace"``)
            to namespace agent-visible files under a single subtree.
        timeout: Default command timeout in seconds for ``execute()``.
    """

    def __init__(
        self,
        workspace: Workspace,
        workdir: str = "/",
        timeout: int = 300,
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    ) -> None:
        self._workspace = workspace
        normalized = workdir.rstrip("/")
        self._workdir: str = normalized or "/"
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._env_cache: SandboxEnvironment | None = None

    @property
    def workspace(self) -> Workspace:
        """The underlying Mirage Workspace instance."""
        return self._workspace

    @property
    def workdir(self) -> str:
        """The virtual root path."""
        return self._workdir

    # --- Path resolution ---

    def _resolve(self, path: str) -> str:
        """Resolve a virtual path to an absolute Mirage workspace path.

        Mirrors :class:`OSBackend` and :class:`AgentFSBackend` semantics
        so backends are interchangeable for the LLM (whose tool prompts
        instruct it to pass *absolute* paths). Traversal escapes raise
        :class:`PermissionError`; the surrounding method translates that
        to ``permission_denied``.

        Detection works in two layers:

        - **`..` outside the workdir:** ``posixpath.normpath`` on the
          stripped path leaves leading ``..`` segments intact only when
          they would escape the root. We reject those explicitly because
          when ``workdir == "/"`` the prefix-comparison check below is
          degenerate (every absolute path "starts with `/`" after
          normalization collapses ``..`` against the root).
        - **Resolved path outside `workdir`:** for non-root workdirs,
          confirm the candidate sits under ``workdir/``.
        """
        if not path:
            return self._workdir

        cleaned = path.lstrip("/")
        normalized_rel = posixpath.normpath(cleaned) if cleaned else "."

        # Leading ".." after normalization → escape attempt against the
        # virtual root. Catches "/../../etc/passwd" regardless of workdir.
        if normalized_rel == ".." or normalized_rel.startswith("../"):
            raise PermissionError(f"Path traversal blocked: {path}")

        if self._workdir == "/":
            if normalized_rel in (".", ""):
                return "/"
            return "/" + normalized_rel

        # Absolute path already under workdir → pass through unchanged.
        # Mirrors DaytonaBackend's branch so the LLM (which is told to
        # use absolute paths and sees ``workdir`` in the ``<env>`` block)
        # doesn't end up writing to ``/workspace/workspace/foo`` when it
        # follows the prompt and emits ``/workspace/foo``.
        root_prefix = self._workdir + "/"
        if path.startswith("/"):
            absolute_normalized = posixpath.normpath(path)
            if absolute_normalized == self._workdir or absolute_normalized.startswith(root_prefix):
                return absolute_normalized
            # Absolute path NOT under workdir → fall through and treat the
            # leading slash as workdir-rooted (legacy ``/foo`` shorthand).

        if normalized_rel in (".", ""):
            return self._workdir

        candidate = posixpath.normpath(f"{self._workdir}/{normalized_rel}")
        if candidate != self._workdir and not candidate.startswith(root_prefix):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    async def _ensure_parents(self, path: str) -> None:
        """Create parent directories of ``path`` (mkdir -p semantics).

        Mirage's ``ws.ops.mkdir`` is single-level — calling it on
        ``/a/b/c`` when ``/a`` doesn't exist raises ``FileNotFoundError``.
        We walk the path components from root toward target, creating
        each in turn and tolerating ``FileExistsError`` for components
        that already exist. ``OSBackend`` gets this for free via
        ``os.makedirs(..., exist_ok=True)``.
        """
        if not path or path == "/":
            return
        parts = path.strip("/").split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}"
            with contextlib.suppress(FileExistsError):
                await self._workspace.ops.mkdir(current)

    # --- FilesystemProtocol: file ops via ws.ops ---

    async def read_bytes(self, path: str) -> ReadBytesResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadBytesResult(error="permission_denied", error_message=str(exc))
        try:
            content = await self._workspace.ops.read(real_path)
        except Exception as exc:  # noqa: BLE001
            return ReadBytesResult(error=_map_exc(exc), error_message=str(exc))
        return ReadBytesResult(content=content)

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a text file with line-based slicing.

        Mirage's ``ws.ops.read`` returns raw ``bytes``; AgentKit's
        protocol contract is *line*-based offset/limit (see
        :meth:`OSBackend.read`), so we read the whole file, decode UTF-8,
        split on line boundaries (preserving newlines), and slice the
        requested window. ``UnicodeDecodeError`` maps to
        ``decode_error`` so the LLM is told to use ``read_bytes`` for
        binary content.
        """
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadResult(error="permission_denied", error_message=str(exc))
        try:
            raw = await self._workspace.ops.read(real_path)
        except Exception as exc:  # noqa: BLE001
            return ReadResult(error=_map_exc(exc), error_message=str(exc))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ReadResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )
        lines = text.splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        return ReadResult(content="".join(selected))

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return WriteResult(error="permission_denied", error_message=str(exc))
        raw = content.encode("utf-8") if isinstance(content, str) else content
        parent = posixpath.dirname(real_path)
        try:
            await self._ensure_parents(parent)
            await self._workspace.ops.write(real_path, raw)
        except Exception as exc:  # noqa: BLE001
            return WriteResult(error=_map_exc(exc), error_message=str(exc))
        return WriteResult(path=path, bytes_written=len(raw))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace ``old_string`` with ``new_string`` in a text file.

        Implemented at the wrapper level — Mirage has no native edit
        primitive, so we read+count+replace+write. Mirrors
        :meth:`OSBackend.edit` semantics exactly so the LLM-actionable
        error vocabulary (``old_string_not_found``, ``ambiguous_match``)
        is consistent across backends.
        """
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return EditResult(error="permission_denied", error_message=str(exc))
        try:
            raw = await self._workspace.ops.read(real_path)
        except Exception as exc:  # noqa: BLE001
            return EditResult(error=_map_exc(exc), error_message=str(exc))
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return EditResult(
                error="decode_error",
                error_message=f"File is not valid UTF-8: {path} ({exc.reason})",
            )

        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(
                path=path,
                error="old_string_not_found",
                error_message=f"old_string not found in {path}",
            )
        if occurrences > 1 and not replace_all:
            return EditResult(
                path=path,
                occurrences=occurrences,
                error="ambiguous_match",
                error_message=(
                    f"old_string appears {occurrences} times in {path}. "
                    "Pass replace_all=True or extend old_string for uniqueness."
                ),
            )
        new_content = (
            content.replace(old_string, new_string)
            if replace_all
            else content.replace(old_string, new_string, 1)
        )
        replacements = occurrences if replace_all else 1
        try:
            await self._workspace.ops.write(real_path, new_content.encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return EditResult(error=_map_exc(exc), error_message=str(exc))
        return EditResult(path=path, replacements=replacements)

    async def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching ``pattern`` via Mirage's built-in ``find``.

        Recursive patterns (``**/*.py``) are translated to ``find -path``
        predicates; flat patterns (``*.py``) use ``find -name``. Matches
        :meth:`DaytonaBackend.glob` so the LLM gets identical results
        regardless of which backend is wired in.
        """
        try:
            real_path = self._resolve(path)
        except PermissionError:
            return []
        if "**" in pattern or "/" in pattern:
            find_pattern = pattern.replace("**", "*")
            full_pattern = f"{real_path.rstrip('/')}/{find_pattern}"
            cmd = (
                f"find {_shell_quote(real_path)}"
                f" -path {_shell_quote(full_pattern)}"
                f" -type f 2>/dev/null | sort"
            )
        else:
            cmd = (
                f"find {_shell_quote(real_path)}"
                f" -name {_shell_quote(pattern)}"
                f" -type f 2>/dev/null | sort"
            )
        io = await self._workspace.execute(cmd)
        output = await io.stdout_str()
        if not output.strip():
            return []
        matches: list[str] = []
        for line in output.strip().splitlines():
            rel = line.removeprefix(self._workdir) if self._workdir != "/" else line
            if not rel.startswith("/"):
                rel = "/" + rel
            matches.append(rel)
        return matches

    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]:
        """Search file contents for ``pattern`` via Mirage's ``grep``.

        Mirage's grep follows POSIX exit-code semantics: ``0`` if at
        least one match, ``1`` if none. Either case is non-fatal here —
        we parse stdout regardless and return an empty list when there
        are no lines, which is what the protocol requires.
        """
        try:
            real_path = self._resolve(path or "/")
        except PermissionError:
            return []
        flags = "-rn"
        if ignore_case:
            flags += "i"
        cmd = f"grep {flags} {_shell_quote(pattern)} {_shell_quote(real_path)}"
        if glob:
            cmd += f" --include={_shell_quote(glob)}"
        cmd += " 2>/dev/null || true"
        io = await self._workspace.execute(cmd)
        output = await io.stdout_str()
        if not output.strip():
            return []
        matches: list[GrepMatch] = []
        for line in output.strip().splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path = parts[0].removeprefix(self._workdir) if self._workdir != "/" else parts[0]
            if not file_path.startswith("/"):
                file_path = "/" + file_path
            try:
                line_num = int(parts[1])
            except ValueError:
                continue
            matches.append(GrepMatch(path=file_path, line=line_num, text=parts[2]))
        return matches

    # --- SandboxProtocol ---

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Run a command in the Mirage workspace.

        Mirage parses the command via tree-sitter-bash and dispatches to
        in-process command handlers — no real shell is spawned. The
        supported command set is whatever each mounted resource has
        registered (see :meth:`environment`). Unsupported binaries
        (``make``, ``python``, ``git``) surface a non-zero exit code with
        a stderr message.

        Output handling mirrors :class:`OSBackend.execute`: stdout and
        stderr are funneled through :class:`BoundedCapture` so a
        runaway pipeline produces a tail window plus a temp-file
        spillover, not a context-blowing dump. Mirage materializes the
        full ``IOResult`` before we feed the bytes in (no streaming),
        which means interleaving order in the spill file is
        ``stdout-then-stderr`` rather than the original write order —
        Mirage already loses the original interleaving by the time we
        receive it.

        Timeouts are enforced via ``asyncio.wait_for`` plus the
        ``cancel`` :class:`asyncio.Event` Mirage exposes; on expiry the
        cancel signal fires and we return ``exit_code=-1`` with
        ``truncated=True``.
        """
        try:
            cwd = self._resolve(workdir) if workdir else self._workdir
        except PermissionError as exc:
            return ExecuteResponse(
                output="",
                stderr=str(exc),
                exit_code=-1,
                truncated=False,
                output_path=None,
                lines_dropped=0,
                bytes_dropped=0,
            )
        effective_timeout = timeout if timeout is not None else self._timeout
        cancel = asyncio.Event()
        capture = BoundedCapture(
            stdout_max_bytes=self._max_output_bytes,
            stdout_max_lines=self._max_output_lines,
            stderr_max_bytes=self._max_output_bytes,
            stderr_max_lines=self._max_output_lines,
        )
        try:
            try:
                io = await asyncio.wait_for(
                    self._workspace.execute(command, cwd=cwd, cancel=cancel),
                    timeout=effective_timeout,
                )
            except TimeoutError:
                cancel.set()
                capture.abandon()
                return ExecuteResponse(
                    output="",
                    stderr=f"Command timed out after {effective_timeout}s",
                    exit_code=-1,
                    truncated=True,
                    output_path=None,
                    lines_dropped=0,
                    bytes_dropped=0,
                )

            stdout_text = await io.stdout_str(errors="replace")
            stderr_text = await io.stderr_str(errors="replace")
            await capture.feed_stdout(stdout_text.encode("utf-8", errors="replace"))
            await capture.feed_stderr(stderr_text.encode("utf-8", errors="replace"))
            stdout_res, stderr_res, spill_path = capture.finalize()
            return ExecuteResponse(
                output=stdout_res.tail.decode("utf-8", errors="replace"),
                stderr=stderr_res.tail.decode("utf-8", errors="replace"),
                exit_code=io.exit_code,
                truncated=spill_path is not None,
                output_path=str(spill_path) if spill_path is not None else None,
                lines_dropped=stdout_res.lines_dropped + stderr_res.lines_dropped,
                bytes_dropped=stdout_res.bytes_dropped + stderr_res.bytes_dropped,
            )
        except Exception:
            capture.abandon()
            raise

    async def environment(self) -> SandboxEnvironment:
        """Snapshot the Mirage workspace environment.

        Unlike OS/Daytona/Bubblewrap (which probe ``PROBED_TOOLS``
        against a real shell) Mirage's command vocabulary is the union
        of every mounted resource's registered commands. We expose that
        union as ``available_tools`` so the LLM, looking at the
        ``<env>`` block, can see the actual tools it has access to —
        including resource-specific commands a remote mount adds (e.g.,
        a Slack mount might register ``slack-search``).

        ``shell`` is set to ``"mirage-bash"`` as an explicit honesty
        signal: this is bash-grammar (tree-sitter-parsed) but a curated
        subset of POSIX tools, not a real shell. An LLM following its
        prompt should consult ``available_tools`` before reaching for
        ``make`` or ``python``.

        Cached after the first call — Mirage's mount set may change via
        ``mount()``/``unmount()`` but the workspace passed in is
        treated as static for the backend's lifetime; callers who hot-
        swap mounts should construct a new backend.
        """
        if self._env_cache is not None:
            return self._env_cache
        tools: set[str] = set()
        for mount in self._workspace.mounts():
            with contextlib.suppress(Exception):
                tools.update(mount.commands().keys())
        self._env_cache = SandboxEnvironment(
            os=f"{platform.system()} {platform.release()} {platform.machine()}",
            shell="mirage-bash",
            cwd=self._workdir,
            available_tools=frozenset(tools),
        )
        return self._env_cache

    # --- FilesystemProtocol: bulk transfer (host-side) ---
    #
    # Mirage has no native bulk-transfer primitive at the workspace
    # level. We loop over ``ws.ops.read``/``write`` and aggregate per-
    # file results — same shape as :meth:`OSBackend.upload` /
    # :meth:`OSBackend.download`. Sequential is fine here: typical
    # seeding flows (``read_tree``-style) move tens of files, not
    # thousands. If a future Mirage release adds bulk endpoints we can
    # switch the implementation without changing the protocol contract.

    async def upload(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]:
        results: list[FileUploadResult] = []
        for path, content in files:
            try:
                real_path = self._resolve(path)
            except PermissionError as exc:
                results.append(
                    FileUploadResult(
                        path=path,
                        error="permission_denied",
                        error_message=str(exc),
                    )
                )
                continue
            try:
                await self._ensure_parents(posixpath.dirname(real_path))
                await self._workspace.ops.write(real_path, content)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    FileUploadResult(
                        path=path,
                        error=_map_exc(exc),
                        error_message=str(exc),
                    )
                )
                continue
            results.append(FileUploadResult(path=path, bytes_written=len(content)))
        return results

    async def download(self, paths: list[str]) -> list[FileDownloadResult]:
        results: list[FileDownloadResult] = []
        for path in paths:
            try:
                real_path = self._resolve(path)
            except PermissionError as exc:
                results.append(
                    FileDownloadResult(
                        path=path,
                        error="permission_denied",
                        error_message=str(exc),
                    )
                )
                continue
            try:
                content = await self._workspace.ops.read(real_path)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    FileDownloadResult(
                        path=path,
                        error=_map_exc(exc),
                        error_message=str(exc),
                    )
                )
                continue
            results.append(FileDownloadResult(path=path, content=content))
        return results
