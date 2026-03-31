"""Permission system for gating tool access.

Two gates:
    1. **Registration** — ``deny`` operations → tool removed entirely.
    2. **Per-call** — path/command checked against rules:
       ``allow`` → execute, ``deny`` → error, ``ask`` → HITL or deny.

Usage::

    from langchain_agentkit.permissions import DEFAULT_RULESET, check_permission

    action = check_permission(DEFAULT_RULESET, "execute", "rm -rf /tmp/data")
"""

from langchain_agentkit.permissions.presets import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
)
from langchain_agentkit.permissions.types import (
    DANGEROUS_COMMANDS,
    SECRETS_PATTERNS,
    OperationPermissions,
    PermissionAction,
    PermissionOperation,
    PermissionRule,
    PermissionRuleset,
    check_permission,
)

__all__ = [
    # Types
    "OperationPermissions",
    "PermissionAction",
    "PermissionOperation",
    "PermissionRule",
    "PermissionRuleset",
    # Checker
    "check_permission",
    # Presets
    "DEFAULT_RULESET",
    "PERMISSIVE_RULESET",
    "READONLY_RULESET",
    "STRICT_RULESET",
    # Constants
    "DANGEROUS_COMMANDS",
    "SECRETS_PATTERNS",
]
