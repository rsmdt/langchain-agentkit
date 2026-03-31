"""Permission presets for common use cases.

Four presets covering the spectrum from read-only to fully permissive:

    READONLY_RULESET    — read + search only, deny all mutations
    DEFAULT_RULESET     — read freely, ask for writes/execute, deny secrets
    PERMISSIVE_RULESET  — allow everything except secrets and dangerous commands
    STRICT_RULESET      — ask for everything, deny secrets

Usage::

    from langchain_agentkit.permissions import DEFAULT_RULESET

    ext = FilesystemExtension(backend=my_backend, permissions=DEFAULT_RULESET)
"""

from langchain_agentkit.permissions.types import (
    DANGEROUS_COMMANDS,
    SECRETS_PATTERNS,
    OperationPermissions,
    PermissionRule,
    PermissionRuleset,
)

_DENY_SECRETS = [
    PermissionRule(pattern=p, action="deny", description="Protect sensitive files")
    for p in SECRETS_PATTERNS
]

_DENY_DANGEROUS = [
    PermissionRule(pattern=p, action="deny", description="Block dangerous command")
    for p in DANGEROUS_COMMANDS
]


READONLY_RULESET = PermissionRuleset(
    default="deny",
    read=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="allow"),
    grep=OperationPermissions(default="allow"),
    write=OperationPermissions(default="deny"),
    edit=OperationPermissions(default="deny"),
    execute=OperationPermissions(default="deny"),
)
"""Read + search only. Deny all mutations and shell execution."""

DEFAULT_RULESET = PermissionRuleset(
    default="ask",
    read=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="allow"),
    grep=OperationPermissions(default="allow"),
    write=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    edit=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    execute=OperationPermissions(default="ask", rules=list(_DENY_DANGEROUS)),
)
"""Read freely. Ask for writes and execution. Deny secrets and dangerous commands."""

PERMISSIVE_RULESET = PermissionRuleset(
    default="allow",
    read=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="allow"),
    grep=OperationPermissions(default="allow"),
    write=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    edit=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    execute=OperationPermissions(default="allow", rules=list(_DENY_DANGEROUS)),
)
"""Allow everything except secrets and dangerous commands."""

STRICT_RULESET = PermissionRuleset(
    default="ask",
    read=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="ask"),
    grep=OperationPermissions(default="ask"),
    write=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    edit=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    execute=OperationPermissions(default="ask", rules=list(_DENY_DANGEROUS)),
)
"""Ask for everything. Deny secrets and dangerous commands."""
