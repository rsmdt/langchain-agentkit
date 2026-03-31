"""FilesystemExtension — Claude Code-aligned file tools for LangGraph agents.

Tools: Read, Write, Edit, Glob, Grep, and optionally Bash.
Gated by a ``PermissionRuleset`` with two enforcement gates:

1. **Registration** — ``deny`` operations have their tools removed entirely.
2. **Per-call** — each invocation checks the specific path/command:
   - ``allow`` → execute immediately
   - ``deny`` → deny with error message
   - ``ask`` + HITL → interrupt for human approval
   - ``ask`` - HITL → deny with actionable error message

Usage::

    from langchain_agentkit import FilesystemExtension
    from langchain_agentkit.permissions import DEFAULT_RULESET

    # Permissive — no gating
    ext = FilesystemExtension(backend=my_backend)

    # Gated — Bash requires approval
    ext = FilesystemExtension(backend=my_backend, permissions=DEFAULT_RULESET)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.backends.protocol import BackendProtocol
from langchain_agentkit.extension import Extension
from langchain_agentkit.permissions.types import (
    PermissionOperation,
    PermissionRuleset,
    check_permission,
)
from langchain_agentkit.extensions.filesystem.tools import create_filesystem_tools

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime

logger = logging.getLogger(__name__)

# Maps tool names to permission operations
_TOOL_OPERATION_MAP: dict[str, PermissionOperation] = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Glob": "glob",
    "Grep": "grep",
    "Bash": "execute",
}

# Maps tool names to the argument that contains the target path/command
_TOOL_TARGET_ARG: dict[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "Bash": "command",
}


class FilesystemExtension(Extension):
    """Extension providing Claude Code-aligned filesystem tools.

    Tools: Read, Write, Edit, Glob, Grep. When the backend supports
    ``execute()``, a Bash tool is also included (gated by permissions).

    Permission enforcement:
        - Gate 1 (registration): ``deny`` → tool removed from toolset.
        - Gate 2 (per-call): ``allow``/``deny``/``ask`` checked per path.
        - ``ask`` without HITL → denied with actionable error message.

    HITL detection: if ``HITLExtension`` is a sibling extension in the
    same ``AgentKit``, ``ask`` permissions trigger ``interrupt()``.
    Otherwise, ``ask`` degrades to ``deny`` with a message telling the
    model why and how to fix it.

    Args:
        backend: A ``BackendProtocol`` implementation.
        root: Root directory for the default OS filesystem backend.
        permissions: Optional permission ruleset. When ``None``, all
            operations are allowed (no gating).
    """

    def __init__(
        self,
        backend: BackendProtocol | None = None,
        root: str | Path = ".",
        permissions: PermissionRuleset | None = None,
    ) -> None:
        self._backend = backend or OSBackend(str(root))
        self._permissions = permissions
        self._hitl_available = False  # Resolved by AgentKit._wire_extensions
        self._tools_cache: list[BaseTool] | None = None

    @property
    def backend(self) -> BackendProtocol:
        """The backend this extension operates on."""
        return self._backend

    @property
    def permissions(self) -> PermissionRuleset | None:
        """The permission ruleset, or None if ungated."""
        return self._permissions

    def set_hitl_available(self, available: bool) -> None:
        """Set whether HITL (interrupt-based approval) is available.

        Called by AgentKit._wire_extensions when HITLExtension is detected
        as a sibling extension.
        """
        self._hitl_available = available

    @property
    def state_schema(self) -> None:
        """No additional state keys."""
        return None

    @property
    def tools(self) -> list[BaseTool]:
        """Filesystem tools, gated by permissions."""
        if self._tools_cache is None:
            self._tools_cache = self._build_tools()
        return self._tools_cache

    def _build_tools(self) -> list[BaseTool]:
        """Build tool list with permission gates applied."""
        tools: list[BaseTool] = create_filesystem_tools(self._backend)

        # Add Bash tool if backend supports execute
        if hasattr(self._backend, "execute") and callable(self._backend.execute):
            tools.append(_build_bash_tool(self._backend))

        if self._permissions is None:
            return tools

        # Gate 1: Remove tools whose operation is fully denied
        filtered: list[BaseTool] = []
        for tool in tools:
            operation = _TOOL_OPERATION_MAP.get(tool.name)
            if operation is None:
                filtered.append(tool)
                continue
            op_perms = self._permissions.get_operation(operation)
            if op_perms.default == "deny" and not op_perms.rules:
                # Fully denied — no rules could override, remove entirely
                logger.debug("Tool '%s' removed: operation '%s' denied", tool.name, operation)
                continue
            filtered.append(tool)

        # Gate 2: Wrap tools that need per-call permission checking
        wrapped: list[BaseTool] = []
        for tool in filtered:
            operation = _TOOL_OPERATION_MAP.get(tool.name)
            if operation is None:
                wrapped.append(tool)
                continue
            wrapped.append(
                _wrap_with_permission_check(
                    tool=tool,
                    operation=operation,
                    target_arg=_TOOL_TARGET_ARG.get(tool.name, ""),
                    permissions=self._permissions,
                    hitl_check=lambda: self._hitl_available,
                )
            )

        return wrapped

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
    ) -> str | None:
        """Return filesystem prompt section."""
        root = ""
        if isinstance(self._backend, OSBackend):
            root = f" rooted at `{self._backend._root}`"
        has_bash = any(t.name == "Bash" for t in self.tools)
        tools_str = "Read, Write, Edit, Glob, Grep"
        if has_bash:
            tools_str += ", Bash"
        return (
            f"## Filesystem\n\n"
            f"You have access to a filesystem{root}.\n\n"
            f"Use the {tools_str} tools to interact with files."
        )


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


class _BashInput(BaseModel):
    command: str = Field(
        description="The shell command to execute.",
    )
    timeout: int | None = Field(
        default=None,
        description="Max seconds to wait. None = no limit.",
    )


def _build_bash_tool(backend: BackendProtocol) -> BaseTool:
    """Build the Bash tool for shell command execution."""

    def _bash(command: str, timeout: int | None = None) -> str:
        result = backend.execute(command, timeout=timeout)
        output = result.get("output", "")
        exit_code = result.get("exit_code", -1)
        if result.get("truncated"):
            output += "\n... (output truncated)"
        if exit_code != 0:
            return f"Exit code {exit_code}\n{output}"
        return output

    return StructuredTool.from_function(
        func=_bash,
        name="Bash",
        description=(
            "Execute a shell command and return its output. "
            "Use for system commands, running scripts, installing packages, "
            "or any operation that requires a shell."
        ),
        args_schema=_BashInput,
        handle_tool_error=True,
    )


# ---------------------------------------------------------------------------
# Permission wrapper
# ---------------------------------------------------------------------------


def _wrap_with_permission_check(
    tool: BaseTool,
    operation: PermissionOperation,
    target_arg: str,
    permissions: PermissionRuleset,
    hitl_check: Any,
) -> BaseTool:
    """Wrap a tool with per-call permission checking.

    Returns a new StructuredTool that checks permissions before
    delegating to the original tool.
    """
    original_func = tool.func

    def _checked(**kwargs: Any) -> Any:
        target = kwargs.get(target_arg, "*")
        action = check_permission(permissions, operation, str(target))

        if action == "allow":
            return original_func(**kwargs)

        if action == "deny":
            raise ToolException(
                f"Permission denied: {operation} on '{target}' is not allowed "
                f"by the configured permission ruleset."
            )

        # action == "ask"
        if hitl_check():
            # HITL available — use LangGraph interrupt for human approval
            from langgraph.types import interrupt

            decision = interrupt({
                "operation": operation,
                "target": target,
                "tool": tool.name,
                "message": f"Agent wants to {operation} '{target}'. Approve?",
            })

            decision_type = (
                decision.get("type", "reject")
                if isinstance(decision, dict)
                else "reject"
            )

            if decision_type == "approve":
                return original_func(**kwargs)

            reason = (
                decision.get("message", "User rejected the operation")
                if isinstance(decision, dict)
                else "User rejected the operation"
            )
            raise ToolException(f"Operation rejected: {reason}")

        # No HITL — deny with actionable message
        raise ToolException(
            f"Permission required: {operation} on '{target}' needs human approval, "
            f"but no approval mechanism is configured. "
            f"Add HITLExtension to the agent's extensions list to enable "
            f"interactive approvals for this operation."
        )

    return StructuredTool.from_function(
        func=_checked,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        handle_tool_error=True,
    )
