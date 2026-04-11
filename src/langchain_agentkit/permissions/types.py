"""Permission types for gating tool access.

Two gates:
    1. **Registration** — ``deny`` operations have their tools removed entirely.
    2. **Per-call** — each invocation checks the specific path/command:
       - ``allow`` → execute immediately
       - ``deny`` → deny with error message
       - ``ask`` + HITL → interrupt for human approval
       - ``ask`` - HITL → deny with actionable error message
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

PermissionAction = Literal["allow", "deny", "ask"]
PermissionOperation = Literal["read", "write", "edit", "execute", "glob", "grep"]

# Patterns matching sensitive files that should never be accessible
SECRETS_PATTERNS: list[str] = [
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/*.crt",
    "**/credentials*",
    "**/secrets*",
    "**/*secret*",
    "**/*password*",
    "**/.aws/**",
    "**/.ssh/**",
    "**/.gnupg/**",
]

# Shell commands that should never be executed
DANGEROUS_COMMANDS: list[str] = [
    "rm -rf /*",
    "rm -rf /",
    ":(){ :|:& };:",  # fork bomb
    "dd if=*of=/dev/*",
    "mkfs.*",
    "> /dev/sda",
    "chmod -R 777 /",
]


@dataclass(frozen=True)
class PermissionRule:
    """A single permission rule matching a path or command pattern.

    Patterns use fnmatch-style globs with ``**`` for recursive matching.
    Rules are evaluated in order — first match wins.

    Args:
        pattern: fnmatch glob pattern (e.g., ``**/.env*``, ``/tmp/**``).
        action: What to do when the pattern matches.
        description: Human-readable reason for the rule.
    """

    pattern: str
    action: PermissionAction
    description: str = ""


@dataclass
class OperationPermissions:
    """Permission configuration for a single operation.

    Args:
        default: Fallback action when no rule matches.
        rules: Ordered list of rules. First match wins.
    """

    default: PermissionAction = "allow"
    rules: list[PermissionRule] = field(default_factory=list)


@dataclass
class PermissionRuleset:
    """Complete permission configuration for all operations.

    Each operation can have its own ``OperationPermissions`` with
    rules and a default. The global ``default`` applies when an
    operation has no specific configuration.

    Args:
        default: Global fallback action.
        read: Permissions for read operations.
        write: Permissions for write operations.
        edit: Permissions for edit operations.
        execute: Permissions for shell execution.
        glob: Permissions for glob operations.
        grep: Permissions for grep operations.
    """

    default: PermissionAction = "ask"
    read: OperationPermissions | None = None
    write: OperationPermissions | None = None
    edit: OperationPermissions | None = None
    execute: OperationPermissions | None = None
    glob: OperationPermissions | None = None
    grep: OperationPermissions | None = None

    def get_operation(self, operation: PermissionOperation) -> OperationPermissions:
        """Get permissions for an operation, falling back to global default."""
        specific = getattr(self, operation, None)
        if isinstance(specific, OperationPermissions):
            return specific
        return OperationPermissions(default=self.default)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a fnmatch-style glob with ``**`` support to a regex.

    - ``*`` matches anything except ``/``
    - ``**`` matches anything including ``/`` (recursive)
    - ``?`` matches any single character except ``/``
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # ** — match anything including /
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    parts.append("(?:.*/)?")
                    i += 3
                else:
                    parts.append(".*")
                    i += 2
            else:
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        elif c in ".+^${}()|[]":
            parts.append(f"\\{c}")
            i += 1
        else:
            parts.append(c)
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def check_permission(
    ruleset: PermissionRuleset,
    operation: PermissionOperation,
    target: str,
) -> PermissionAction:
    """Check permission for an operation on a target.

    Evaluates rules in order. First match wins. Falls back to
    operation default, then global default.

    Args:
        ruleset: The permission configuration.
        operation: The operation being attempted.
        target: The path or command being operated on.

    Returns:
        The permission action: ``"allow"``, ``"deny"``, or ``"ask"``.
    """
    op_perms = ruleset.get_operation(operation)
    for rule in op_perms.rules:
        regex = _glob_to_regex(rule.pattern)
        if regex.match(target):
            return rule.action
    return op_perms.default
