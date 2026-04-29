"""DaytonaBackend — Daytona cloud sandbox backend.

Implements ``BackendProtocol``, ``SandboxBackend``, and
``FileTransferBackend`` using the Daytona SDK for ``execute()`` and
shell commands for file operations (read, write, edit, glob, grep).

Requires ``pip install langchain-agentkit[daytona]`` (or
``pip install daytona-sdk`` directly). Importing this module fails with
a clear ``ImportError`` if the SDK isn't installed — DaytonaBackend
isn't re-exported from the generic ``langchain_agentkit.backends``
namespace, so SDK-less users running OSBackend are unaffected.

Usage::

    from daytona_sdk import Daytona, DaytonaConfig
    from langchain_agentkit.backends.daytona import DaytonaBackend

    config = DaytonaConfig(api_key="...", api_url="http://localhost:3000")
    sandbox = Daytona(config).create()
    backend = DaytonaBackend(sandbox)

    result = await backend.read("/app/main.py")
    if result.error is None:
        print(result.content)
"""

from __future__ import annotations

import base64
import json
import posixpath
from typing import TYPE_CHECKING

from daytona_sdk import FileDownloadRequest, FileUpload

from langchain_agentkit.backends.protocol import (
    ExecuteResponse,
    GrepMatch,
)
from langchain_agentkit.backends.results import (
    PROBED_TOOLS,
    EditResult,
    FileDownloadResult,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    WriteResult,
)

if TYPE_CHECKING:
    from daytona_sdk import Sandbox


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation (POSIX)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# Single-shot environment probe. Output layout (line-oriented):
#   line 1:  uname -srm output (e.g. "Linux 5.15.0-91-generic x86_64")
#   line 2:  $SHELL value (or "/bin/sh" if unset)
#   line 3:  literal separator "TOOLS_BEGIN"
#   lines 4+: one available tool name per line, in PROBED_TOOLS order
#
# Daytona's ``process.exec`` runs commands under zsh by default; ``===``
# trips zsh's ``=cmd`` path-expansion syntax, so the separator is an
# unambiguous identifier instead.
_ENV_PROBE_SEPARATOR = "TOOLS_BEGIN"
_ENV_PROBE_SCRIPT = (
    "uname -srm\n"
    'echo "${SHELL:-/bin/sh}"\n'
    f"echo {_ENV_PROBE_SEPARATOR}\n"
    "for t in " + " ".join(PROBED_TOOLS) + "; do "
    'command -v "$t" >/dev/null 2>&1 && echo "$t"; '
    "done"
)


# Python program executed inside the sandbox to perform an edit.
# Emits a single JSON line on stdout with the result. Status keys:
#   error: "file_not_found" | "is_directory" | "decode_error" |
#          "old_string_not_found" | "ambiguous_match" | "io_error"
#   occurrences: int  (when error == "ambiguous_match")
#   message: str      (when error == "io_error")
#   replacements: int (on success)
_EDIT_SCRIPT = """
import json, base64, sys
d = json.loads(base64.b64decode(sys.argv[1]))
p = d['path']
try:
    t = open(p).read()
except FileNotFoundError:
    print(json.dumps({'error': 'file_not_found'})); sys.exit(0)
except IsADirectoryError:
    print(json.dumps({'error': 'is_directory'})); sys.exit(0)
except UnicodeDecodeError:
    print(json.dumps({'error': 'decode_error'})); sys.exit(0)
except OSError as e:
    print(json.dumps({'error': 'io_error', 'message': str(e)})); sys.exit(0)
n = t.count(d['old'])
if n == 0:
    print(json.dumps({'error': 'old_string_not_found'})); sys.exit(0)
if n > 1 and not d['replace_all']:
    print(json.dumps({'error': 'ambiguous_match', 'occurrences': n})); sys.exit(0)
new = t.replace(d['old'], d['new']) if d['replace_all'] else t.replace(d['old'], d['new'], 1)
try:
    open(p, 'w').write(new)
except OSError as e:
    print(json.dumps({'error': 'io_error', 'message': str(e)})); sys.exit(0)
print(json.dumps({'replacements': n if d['replace_all'] else 1}))
"""


class DaytonaBackend:
    """Daytona cloud sandbox backend.

    Wraps a Daytona ``Sandbox`` instance. Implements ``execute()``
    via the Daytona process API. File operations are implemented
    via shell commands routed through ``execute()``.

    Args:
        sandbox: A Daytona ``Sandbox`` instance (already created).
        timeout: Default command timeout in seconds.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        timeout: int = 300,
    ) -> None:
        self._sandbox = sandbox
        self._timeout = timeout
        self._workdir: str = str(sandbox.get_work_dir()).rstrip("/")
        self._env_cache: SandboxEnvironment | None = None

    @property
    def sandbox(self) -> Sandbox:
        """The underlying Daytona Sandbox instance."""
        return self._sandbox

    @property
    def workdir(self) -> str:
        """The sandbox working directory."""
        return self._workdir

    # --- Path resolution ---

    def _resolve(self, path: str) -> str:
        """Resolve a virtual path to an absolute sandbox path.

        Raises ``PermissionError`` for paths that escape the workdir.
        Backend methods catch this and convert to ``permission_denied``.
        """
        cleaned = path.lstrip("/")
        if not cleaned:
            return self._workdir
        candidate = posixpath.normpath(f"{self._workdir}/{cleaned}")
        root_prefix = self._workdir.rstrip("/") + "/"
        if candidate != self._workdir and not candidate.startswith(root_prefix):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    # --- SandboxBackend ---

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the Daytona sandbox.

        Raises:
            RuntimeError: If the Daytona SDK call fails (network, auth, etc.).
        """
        cwd = self._resolve(workdir) if workdir else self._workdir
        try:
            result = self._sandbox.process.exec(
                command,
                cwd=cwd,
                timeout=timeout or self._timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Daytona sandbox execution failed for command: {command!r}"
            ) from exc
        stderr = getattr(result, "stderr", None) or ""
        return ExecuteResponse(
            output=result.result or "",
            exit_code=result.exit_code,
            truncated=False,
            stderr=stderr,
            output_path=None,
            lines_dropped=0,
            bytes_dropped=0,
        )

    async def environment(self) -> SandboxEnvironment:
        """Probe the sandbox once for OS/shell/tool inventory.

        One ``execute()`` call emits a delimited script output that we
        parse into a :class:`SandboxEnvironment`. Cached for the backend's
        lifetime — the sandbox's environment doesn't change mid-session.
        Probe failure falls back to "unknown" / "/bin/sh" / empty tools.
        """
        if self._env_cache is not None:
            return self._env_cache

        result = await self.execute(_ENV_PROBE_SCRIPT)
        os_line = "unknown"
        shell = "/bin/sh"
        tools: set[str] = set()
        if result.get("exit_code") == 0:
            lines = result.get("output", "").splitlines()
            try:
                sep_idx = lines.index(_ENV_PROBE_SEPARATOR)
            except ValueError:
                sep_idx = -1
            if sep_idx >= 2:
                os_line = lines[0].strip() or "unknown"
                shell = lines[1].strip() or "/bin/sh"
                tools = {ln.strip() for ln in lines[sep_idx + 1 :] if ln.strip()}

        self._env_cache = SandboxEnvironment(
            os=os_line,
            shell=shell,
            cwd=self._workdir,
            available_tools=frozenset(tools),
        )
        return self._env_cache

    # --- BackendProtocol: file ops via shell ---

    async def read_bytes(self, path: str) -> ReadBytesResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadBytesResult(error="permission_denied", error_message=str(exc))
        check = await self.execute(f"test -f {_shell_quote(real_path)}")
        if check["exit_code"] != 0:
            # Distinguish "is a directory" from "does not exist"
            isdir = await self.execute(f"test -d {_shell_quote(real_path)}")
            if isdir["exit_code"] == 0:
                return ReadBytesResult(
                    error="is_directory", error_message=f"Path is a directory: {path}"
                )
            return ReadBytesResult(error="file_not_found", error_message=f"File not found: {path}")
        result = await self.execute(f"cat {_shell_quote(real_path)} | base64")
        if result["exit_code"] != 0:
            return ReadBytesResult(
                error="io_error",
                error_message=f"Read failed: {result.get('stderr') or result.get('output')}",
            )
        return ReadBytesResult(content=base64.b64decode(result["output"].strip()))

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return ReadResult(error="permission_denied", error_message=str(exc))
        check = await self.execute(f"test -f {_shell_quote(real_path)}")
        if check["exit_code"] != 0:
            isdir = await self.execute(f"test -d {_shell_quote(real_path)}")
            if isdir["exit_code"] == 0:
                return ReadResult(
                    error="is_directory", error_message=f"Path is a directory: {path}"
                )
            return ReadResult(error="file_not_found", error_message=f"File not found: {path}")
        start = offset + 1
        end = offset + limit
        cmd = f"sed -n '{start},{end}p' {_shell_quote(real_path)}"
        result = await self.execute(cmd)
        if result["exit_code"] != 0:
            return ReadResult(
                error="io_error",
                error_message=f"Read failed: {result.get('stderr') or result.get('output')}",
            )
        return ReadResult(content=result["output"])

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return WriteResult(error="permission_denied", error_message=str(exc))
        raw = content.encode("utf-8") if isinstance(content, str) else content
        encoded = base64.b64encode(raw).decode("ascii")
        cmd = (
            f"mkdir -p $(dirname {_shell_quote(real_path)}) && "
            f"echo {_shell_quote(encoded)} | base64 -d > {_shell_quote(real_path)}"
        )
        result = await self.execute(cmd)
        if result["exit_code"] != 0:
            return WriteResult(
                error="io_error",
                error_message=f"Write failed: {result.get('stderr') or result.get('output')}",
            )
        return WriteResult(path=path, bytes_written=len(raw))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            real_path = self._resolve(path)
        except PermissionError as exc:
            return EditResult(error="permission_denied", error_message=str(exc))
        payload = json.dumps(
            {
                "path": real_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            }
        )
        encoded = base64.b64encode(payload.encode()).decode("ascii")
        script = _EDIT_SCRIPT
        result = await self.execute(f"python3 -c {_shell_quote(script)} {_shell_quote(encoded)}")
        if result["exit_code"] != 0:
            return EditResult(
                error="io_error",
                error_message=f"Edit failed: {result.get('stderr') or result.get('output')}",
            )
        try:
            data = json.loads(result["output"].strip())
        except json.JSONDecodeError:
            return EditResult(
                error="io_error",
                error_message=f"Edit produced invalid output: {result['output']!r}",
            )
        if "error" in data:
            err = data["error"]
            if err == "file_not_found":
                return EditResult(error="file_not_found", error_message=f"File not found: {path}")
            if err == "is_directory":
                return EditResult(
                    error="is_directory", error_message=f"Path is a directory: {path}"
                )
            if err == "decode_error":
                return EditResult(
                    error="decode_error",
                    error_message=f"File is not valid UTF-8: {path}",
                )
            if err == "old_string_not_found":
                return EditResult(
                    path=path,
                    error="old_string_not_found",
                    error_message=f"old_string not found in {path}",
                )
            if err == "ambiguous_match":
                occ = data.get("occurrences")
                return EditResult(
                    path=path,
                    occurrences=occ,
                    error="ambiguous_match",
                    error_message=(
                        f"old_string appears {occ} times in {path}. "
                        "Pass replace_all=True or extend old_string for uniqueness."
                    ),
                )
            return EditResult(
                error="io_error",
                error_message=data.get("message") or f"Edit failed: {err}",
            )
        return EditResult(path=path, replacements=data["replacements"])

    async def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Supports recursive patterns like ``**/*.py`` by translating them
        to ``find -path`` predicates, matching ``OSBackend.glob()`` semantics.
        """
        real_path = self._resolve(path)
        if "**" in pattern or "/" in pattern:
            find_pattern = pattern.replace("**", "*")
            full_pattern = f"{real_path}/{find_pattern}"
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
        result = await self.execute(cmd)
        if result["exit_code"] != 0 or not result["output"].strip():
            return []
        matches = []
        for line in result["output"].strip().splitlines():
            rel = line.removeprefix(self._workdir)
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
        """Search file contents for a regex pattern."""
        real_path = self._resolve(path or "/")
        flags = "-rn"
        if ignore_case:
            flags += "i"
        cmd = f"grep {flags} {_shell_quote(pattern)} {_shell_quote(real_path)}"
        if glob:
            cmd += f" --include={_shell_quote(glob)}"
        cmd += " 2>/dev/null || true"
        result = await self.execute(cmd)
        if not result["output"].strip():
            return []
        matches: list[GrepMatch] = []
        for line in result["output"].strip().splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path = parts[0].removeprefix(self._workdir)
            if not file_path.startswith("/"):
                file_path = "/" + file_path
            try:
                line_num = int(parts[1])
            except ValueError:
                continue
            matches.append(GrepMatch(path=file_path, line=line_num, text=parts[2]))
        return matches

    # --- FileTransferBackend ---
    #
    # True bulk transfer via the Daytona SDK's native multipart endpoints.
    # ``sandbox.fs.upload_files`` posts one multipart HTTP request for N
    # files; ``download`` returns one multipart response. No
    # ARG_MAX ceiling, no per-file round trip.

    async def upload(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]:
        if not files:
            return []
        upfront_errors: list[FileUploadResult] = []
        valid: list[tuple[str, bytes, FileUpload]] = []
        for path, content in files:
            try:
                real_path = self._resolve(path)
            except PermissionError as exc:
                upfront_errors.append(
                    FileUploadResult(path=path, error="permission_denied", error_message=str(exc))
                )
                continue
            valid.append((path, content, FileUpload(source=content, destination=real_path)))

        if not valid:
            return upfront_errors

        try:
            self._sandbox.fs.upload_files([f for _, _, f in valid])
        except Exception as exc:
            return upfront_errors + [
                FileUploadResult(
                    path=path,
                    error="io_error",
                    error_message=f"Bulk upload failed: {exc}",
                )
                for path, _, _ in valid
            ]

        return upfront_errors + [
            FileUploadResult(path=path, bytes_written=len(content)) for path, content, _ in valid
        ]

    async def download(self, paths: list[str]) -> list[FileDownloadResult]:
        if not paths:
            return []
        upfront_errors: list[FileDownloadResult] = []
        real_to_virtual: dict[str, str] = {}
        requests: list[FileDownloadRequest] = []
        for path in paths:
            try:
                real_path = self._resolve(path)
            except PermissionError as exc:
                upfront_errors.append(
                    FileDownloadResult(path=path, error="permission_denied", error_message=str(exc))
                )
                continue
            real_to_virtual[real_path] = path
            requests.append(FileDownloadRequest(source=real_path))

        if not requests:
            return upfront_errors

        try:
            sdk_responses = self._sandbox.fs.download_files(requests)
        except Exception as exc:
            return upfront_errors + [
                FileDownloadResult(
                    path=real_to_virtual[r.source],
                    error="io_error",
                    error_message=f"Bulk download failed: {exc}",
                )
                for r in requests
            ]

        results: list[FileDownloadResult] = list(upfront_errors)
        for resp in sdk_responses:
            virtual_path = real_to_virtual.get(resp.source, resp.source)
            if resp.error:
                # SDK error strings are free-form; map "not found" to the
                # standardized code, otherwise fall through to io_error.
                if "not found" in resp.error.lower():
                    results.append(
                        FileDownloadResult(
                            path=virtual_path,
                            error="file_not_found",
                            error_message=resp.error,
                        )
                    )
                else:
                    results.append(
                        FileDownloadResult(
                            path=virtual_path,
                            error="io_error",
                            error_message=resp.error,
                        )
                    )
            elif isinstance(resp.result, (bytes, bytearray)):
                results.append(FileDownloadResult(path=virtual_path, content=bytes(resp.result)))
            else:
                bad_type = type(resp.result).__name__
                results.append(
                    FileDownloadResult(
                        path=virtual_path,
                        error="io_error",
                        error_message=f"Unexpected download result type: {bad_type}",
                    )
                )
        return results
