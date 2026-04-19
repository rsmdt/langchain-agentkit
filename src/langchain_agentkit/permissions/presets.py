"""Permission presets for common use cases.

Four presets covering the spectrum from read-only to fully permissive:

    READONLY_RULESET    — read + search only, deny all mutations
    DEFAULT_RULESET     — read freely, ask for writes/execute, deny secrets
    PERMISSIVE_RULESET  — allow everything except secrets and dangerous commands
    STRICT_RULESET      — ask for everything, deny secrets

All presets also deny writes/edits to ``.agentkit/**`` — the agent's own
configuration tree — so hosts cannot be accidentally polluted by
self-modification through the bundled Write/Edit/Bash tools. This is a
convention-over-configuration default; construct a custom
``PermissionRuleset`` if a self-editing agent is actually desired.

Usage::

    from langchain_agentkit.permissions import DEFAULT_RULESET

    ext = FilesystemExtension(backend=my_backend, permissions=DEFAULT_RULESET)
"""

from langchain_agentkit.permissions.types import (
    AGENTKIT_COMMAND_PATTERNS,
    AGENTKIT_PATTERNS,
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

_DENY_AGENTKIT = [
    PermissionRule(
        pattern=p,
        action="deny",
        description="Self-modification of AgentKit configuration files is not permitted",
    )
    for p in AGENTKIT_PATTERNS
]

_DENY_AGENTKIT_SHELL = [
    PermissionRule(
        pattern=p,
        action="deny",
        description="Self-modification of AgentKit configuration via shell is not permitted",
    )
    for p in AGENTKIT_COMMAND_PATTERNS
]

_DENY_DANGEROUS = [
    PermissionRule(pattern=p, action="deny", description="Block dangerous command")
    for p in DANGEROUS_COMMANDS
]

# Baseline deny rules applied to every write/edit operation across every preset.
_BASELINE_WRITE_EDIT_DENIES = [*_DENY_AGENTKIT, *_DENY_SECRETS]

# Baseline deny rules applied to every execute operation across every preset.
_BASELINE_EXECUTE_DENIES = [*_DENY_AGENTKIT_SHELL, *_DENY_DANGEROUS]


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
    write=OperationPermissions(default="ask", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    edit=OperationPermissions(default="ask", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    execute=OperationPermissions(default="ask", rules=list(_BASELINE_EXECUTE_DENIES)),
)
"""Read freely. Ask for writes/execution. Deny ``.agentkit/``, secrets, dangerous cmds."""

PERMISSIVE_RULESET = PermissionRuleset(
    default="allow",
    read=OperationPermissions(default="allow", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="allow"),
    grep=OperationPermissions(default="allow"),
    write=OperationPermissions(default="allow", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    edit=OperationPermissions(default="allow", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    execute=OperationPermissions(default="allow", rules=list(_BASELINE_EXECUTE_DENIES)),
)
"""Allow everything except ``.agentkit/``, secrets, and dangerous commands."""

STRICT_RULESET = PermissionRuleset(
    default="ask",
    read=OperationPermissions(default="ask", rules=list(_DENY_SECRETS)),
    glob=OperationPermissions(default="ask"),
    grep=OperationPermissions(default="ask"),
    write=OperationPermissions(default="ask", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    edit=OperationPermissions(default="ask", rules=list(_BASELINE_WRITE_EDIT_DENIES)),
    execute=OperationPermissions(default="ask", rules=list(_BASELINE_EXECUTE_DENIES)),
)
"""Ask for everything. Deny ``.agentkit/``, secrets, and dangerous commands."""
