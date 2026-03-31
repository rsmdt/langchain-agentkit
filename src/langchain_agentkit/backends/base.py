"""BaseSandbox — implement one method, get all six.

Any sandbox provider with shell access subclasses ``BaseSandbox`` and
implements ``execute()``. File operations (read, write, edit, glob, grep)
are provided via shell commands routed through ``execute()``.

Usage::

    from langchain_agentkit.backends import BaseSandbox, ExecuteResponse

    class MySandbox(BaseSandbox):
        def execute(self, command, timeout=None, workdir=None) -> ExecuteResponse:
            # Run command in your sandbox
            result = my_provider.run(command, timeout=timeout)
            return ExecuteResponse(
                output=result.stdout,
                exit_code=result.returncode,
                truncated=False,
            )

    sandbox = MySandbox(workdir="/home/user")
    content = sandbox.read("/app/main.py")  # works via shell
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod

from langchain_agentkit.backends.protocol import (
    EditResult,
    ExecuteResponse,
    GrepMatch,
    WriteResult,
)


class BaseSandbox(ABC):
    """Abstract base for sandbox backends with shell access.

    Implements all ``BackendProtocol`` file operations by constructing
    shell commands and routing them through the single abstract method
    ``execute()``.

    Subclasses may override individual methods for performance (e.g.,
    using a native file API instead of shell for read/write).

    Args:
        workdir: Default working directory for commands. Paths are
            resolved relative to this directory.
    """

    def __init__(self, workdir: str = "/home/user") -> None:
        self._workdir = workdir.rstrip("/")

    @property
    def workdir(self) -> str:
        """The sandbox working directory."""
        return self._workdir

    def _resolve(self, path: str) -> str:
        """Resolve a virtual path to an absolute sandbox path."""
        cleaned = path.lstrip("/")
        if not cleaned:
            return self._workdir
        return f"{self._workdir}/{cleaned}"

    # --- Abstract: the one method providers implement ---

    @abstractmethod
    def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox.

        This is the single method that sandbox providers must implement.
        All other ``BackendProtocol`` methods are built on top of this.

        Args:
            command: Shell command to execute.
            timeout: Max seconds to wait. None = no limit.
            workdir: Working directory override. None = use default.

        Returns:
            ExecuteResponse with output, exit_code, and truncated flag.
        """
        ...

    # --- File operations via shell ---

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file with line numbers, offset, and limit.

        Uses ``sed`` + ``cat -n`` for pagination without loading the
        entire file into memory on the sandbox side.
        """
        real_path = self._resolve(path)
        start = offset + 1
        end = offset + limit
        # sed -n 'start,end p' prints the line range; cat -n adds line numbers
        cmd = f"sed -n '{start},{end}p' {_shell_quote(real_path)} | cat -n"
        result = self.execute(cmd)
        if result["exit_code"] != 0:
            return ""
        # Reformat: cat -n produces "     1\tline", normalize to "{offset+i}\tline"
        lines: list[str] = []
        for raw_line in result["output"].splitlines(keepends=True):
            stripped = raw_line.lstrip()
            _, _, content = stripped.partition("\t")
            lines.append(f"{offset + len(lines) + 1}\t{content}")
        return "".join(lines)

    def write(self, path: str, content: str | bytes) -> WriteResult:
        """Write content to a file, creating parent directories."""
        real_path = self._resolve(path)
        if isinstance(content, bytes):
            encoded = base64.b64encode(content).decode("ascii")
            cmd = (
                f"mkdir -p $(dirname {_shell_quote(real_path)}) && "
                f"echo {_shell_quote(encoded)} | base64 -d > {_shell_quote(real_path)}"
            )
        else:
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            cmd = (
                f"mkdir -p $(dirname {_shell_quote(real_path)}) && "
                f"echo {_shell_quote(encoded)} | base64 -d > {_shell_quote(real_path)}"
            )
        result = self.execute(cmd)
        if result["exit_code"] != 0:
            raise OSError(f"Write failed: {result['output']}")
        size = len(content) if isinstance(content, str) else len(content)
        return WriteResult(path=path, bytes_written=size)

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Find-and-replace in a file with ambiguity checking.

        For files under 50KB, uses an inline Python script via execute().
        """
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
        # Inline Python script for atomic edit with ambiguity check
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
        result = self.execute(f"python3 -c {_shell_quote(script)} {_shell_quote(encoded)}")
        if result["exit_code"] != 0:
            raise ValueError(f"Edit failed: {result['output']}")
        data = json.loads(result["output"].strip())
        if "error" in data:
            raise ValueError(
                f"{data['error']} in {path}. Use replace_all=True or provide more context."
            )
        return EditResult(path=path, replacements=data["replacements"])

    def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern using ``find``."""
        real_path = self._resolve(path)
        cmd = (
            f"find {_shell_quote(real_path)} -name {_shell_quote(pattern)}"
            f" -type f 2>/dev/null | sort"
        )
        result = self.execute(cmd)
        if result["exit_code"] != 0 or not result["output"].strip():
            return []
        matches = []
        for line in result["output"].strip().splitlines():
            rel = line.removeprefix(self._workdir)
            if not rel.startswith("/"):
                rel = "/" + rel
            matches.append(rel)
        return matches

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]:
        """Search file contents for a regex pattern using ``grep``."""
        real_path = self._resolve(path or "/")
        flags = "-rn"
        if ignore_case:
            flags += "i"
        cmd = f"grep {flags} {_shell_quote(pattern)} {_shell_quote(real_path)}"
        if glob:
            cmd += f" --include={_shell_quote(glob)}"
        cmd += " 2>/dev/null || true"
        result = self.execute(cmd)
        if not result["output"].strip():
            return []
        matches: list[GrepMatch] = []
        for line in result["output"].strip().splitlines():
            # Format: /path/to/file:linenum:text
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


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell interpolation."""
    return "'" + s.replace("'", "'\"'\"'") + "'"
