"""DaytonaBackend — Daytona cloud sandbox backend.

Implements the full ``BackendProtocol`` using the Daytona SDK for
``execute()`` and shell commands for file operations (read, write,
edit, glob, grep).

Requires ``pip install daytona-sdk`` (or ``pip install daytona``).

Usage::

    from daytona_sdk import Daytona, DaytonaConfig
    from langchain_agentkit.backends import DaytonaBackend

    config = DaytonaConfig(api_key="...", api_url="http://localhost:3000")
    sandbox = Daytona(config).create()
    backend = DaytonaBackend(sandbox)

    # All BackendProtocol methods are async:
    content = await backend.read("/app/main.py")  # raw text, no line numbers
    result = await backend.execute("python3 -m pytest")
"""

from __future__ import annotations

import base64
import json
import posixpath
from typing import Any

from langchain_agentkit.backends.protocol import (
    EditResult,
    ExecuteResponse,
    GrepMatch,
    WriteResult,
)


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation (POSIX)."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


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
        sandbox: Any,
        timeout: int = 300,
    ) -> None:
        self._sandbox = sandbox
        self._timeout = timeout
        self._workdir: str = str(sandbox.get_work_dir()).rstrip("/")

    @property
    def sandbox(self) -> Any:
        """The underlying Daytona Sandbox instance."""
        return self._sandbox

    @property
    def workdir(self) -> str:
        """The sandbox working directory."""
        return self._workdir

    # --- Path resolution ---

    def _resolve(self, path: str) -> str:
        """Resolve a virtual path to an absolute sandbox path.

        Blocks path traversal: any path that escapes the workdir
        raises ``PermissionError``, matching ``OSBackend._resolve()``.
        """
        cleaned = path.lstrip("/")
        if not cleaned:
            return self._workdir
        candidate = posixpath.normpath(f"{self._workdir}/{cleaned}")
        root_prefix = self._workdir.rstrip("/") + "/"
        if candidate != self._workdir and not candidate.startswith(root_prefix):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    # --- execute() — the Daytona SDK bridge ---

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

    # --- File operations via shell ---

    async def read_bytes(self, path: str) -> bytes:
        """Read a file as raw bytes."""
        real_path = self._resolve(path)
        # Check file existence first (pipe masks cat exit code on some shells)
        check = f"test -f {_shell_quote(real_path)}"
        if (await self.execute(check))["exit_code"] != 0:
            raise FileNotFoundError(f"File not found: {path}")
        # Use cat | base64 for portability (macOS base64 requires -i flag)
        cmd = f"cat {_shell_quote(real_path)} | base64"
        result = await self.execute(cmd)
        if result["exit_code"] != 0:
            raise FileNotFoundError(f"File not found: {path}")
        return base64.b64decode(result["output"].strip())

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read raw text content with offset/limit support."""
        real_path = self._resolve(path)
        # Check file existence first (pipe masks sed exit code on some shells)
        check = f"test -f {_shell_quote(real_path)}"
        if (await self.execute(check))["exit_code"] != 0:
            raise FileNotFoundError(f"File not found: {path}")
        start = offset + 1
        end = offset + limit
        cmd = f"sed -n '{start},{end}p' {_shell_quote(real_path)}"
        result = await self.execute(cmd)
        return result["output"]

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        """Write content to a file, creating parent directories."""
        real_path = self._resolve(path)
        raw = content.encode("utf-8") if isinstance(content, str) else content
        encoded = base64.b64encode(raw).decode("ascii")
        cmd = (
            f"mkdir -p $(dirname {_shell_quote(real_path)}) && "
            f"echo {_shell_quote(encoded)} | base64 -d > {_shell_quote(real_path)}"
        )
        result = await self.execute(cmd)
        if result["exit_code"] != 0:
            raise OSError(f"Write failed: {result['output']}")
        return WriteResult(path=path, bytes_written=len(raw))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Find-and-replace in a file with ambiguity checking."""
        real_path = self._resolve(path)
        payload = json.dumps(
            {
                "path": real_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
            }
        )
        encoded = base64.b64encode(payload.encode()).decode("ascii")
        script = (
            "import json, base64, sys; "
            "d = json.loads(base64.b64decode(sys.argv[1])); "
            "t = open(d['path']).read(); "
            "n = t.count(d['old']); "
            "r = json.dumps({'replacements': 0}) if n == 0 else None; "
            "r = r or ("
            "  json.dumps({'error': f'Ambiguous edit: appears {n} times'}) "
            "  if not d['replace_all'] and n > 1 else None"
            "); "
            "r = r or ("
            "  (open(d['path'], 'w').write("
            "    t.replace(d['old'], d['new']) if d['replace_all'] "
            "    else t.replace(d['old'], d['new'], 1)"
            "  ) and False) or json.dumps({'replacements': n if d['replace_all'] else 1})"
            "); "
            "print(r)"
        )
        result = await self.execute(f"python3 -c {_shell_quote(script)} {_shell_quote(encoded)}")
        if result["exit_code"] != 0:
            raise ValueError(f"Edit failed: {result['output']}")
        data = json.loads(result["output"].strip())
        if "error" in data:
            raise ValueError(
                f"{data['error']} in {path}. Use replace_all=True or provide more context."
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
