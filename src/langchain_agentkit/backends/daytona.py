"""DaytonaSandbox — Daytona cloud sandbox backend.

Extends ``BaseSandbox`` — implements ``execute()`` only. All file
operations (read, write, edit, glob, grep) are inherited via shell
commands.

Requires ``pip install daytona-sdk``.

Usage::

    from daytona_sdk import Daytona
    from langchain_agentkit.backends import DaytonaSandbox

    sandbox = Daytona().create()
    backend = DaytonaSandbox(sandbox)

    # All BackendProtocol methods work:
    content = backend.read("/app/main.py")
    result = backend.execute("python3 -m pytest")
"""

from __future__ import annotations

from typing import Any

from langchain_agentkit.backends.base import BaseSandbox
from langchain_agentkit.backends.protocol import ExecuteResponse


class DaytonaSandbox(BaseSandbox):
    """Daytona cloud sandbox backend.

    Wraps a Daytona ``Sandbox`` instance. Implements ``execute()``
    via the Daytona process API. All file operations are inherited
    from ``BaseSandbox`` (shell-based).

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
        super().__init__(workdir=sandbox.get_work_dir())

    @property
    def sandbox(self) -> Any:
        """The underlying Daytona Sandbox instance."""
        return self._sandbox

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the Daytona sandbox."""
        cwd = self._resolve(workdir) if workdir else self._workdir
        result = self._sandbox.process.exec(
            command,
            cwd=cwd,
            timeout=timeout or self._timeout,
        )
        return ExecuteResponse(
            output=result.result or "",
            exit_code=result.exit_code,
            truncated=False,
        )
