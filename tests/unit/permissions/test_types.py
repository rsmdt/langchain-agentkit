"""Tests for the permission system: types, rules, glob matching, and presets."""

from __future__ import annotations

from langchain_agentkit.permissions.presets import (
    DEFAULT_RULESET,
    PERMISSIVE_RULESET,
    READONLY_RULESET,
    STRICT_RULESET,
)
from langchain_agentkit.permissions.types import (
    OperationPermissions,
    PermissionRule,
    PermissionRuleset,
    _glob_to_regex,
    check_permission,
)

# ---------------------------------------------------------------------------
# _glob_to_regex
# ---------------------------------------------------------------------------


class TestGlobToRegex:
    def test_literal_match(self):
        assert _glob_to_regex("/tmp/file.txt").match("/tmp/file.txt")

    def test_literal_no_match(self):
        assert not _glob_to_regex("/tmp/file.txt").match("/tmp/other.txt")

    def test_single_star_matches_filename(self):
        assert _glob_to_regex("*.py").match("main.py")

    def test_single_star_does_not_cross_slash(self):
        assert not _glob_to_regex("*.py").match("src/main.py")

    def test_double_star_matches_recursive(self):
        assert _glob_to_regex("**/*.py").match("src/deep/main.py")

    def test_double_star_matches_root(self):
        assert _glob_to_regex("**/.env").match(".env")

    def test_double_star_matches_nested(self):
        assert _glob_to_regex("**/.env").match("project/.env")

    def test_double_star_slash_prefix(self):
        assert _glob_to_regex("**/.env.*").match("project/.env.local")

    def test_question_mark(self):
        assert _glob_to_regex("file?.txt").match("file1.txt")
        assert not _glob_to_regex("file?.txt").match("file12.txt")

    def test_dots_escaped(self):
        assert _glob_to_regex("*.env").match("test.env")
        assert not _glob_to_regex("*.env").match("testXenv")

    def test_recursive_directory(self):
        assert _glob_to_regex("/tmp/**").match("/tmp/anything/here")

    def test_double_star_middle(self):
        assert _glob_to_regex("src/**/test.py").match("src/a/b/test.py")


# ---------------------------------------------------------------------------
# check_permission — rule evaluation
# ---------------------------------------------------------------------------


class TestCheckPermission:
    def test_no_rules_returns_default(self):
        ruleset = PermissionRuleset(
            default="deny",
            read=OperationPermissions(default="allow"),
        )
        assert check_permission(ruleset, "read", "/any.txt") == "allow"

    def test_global_default_when_no_operation_config(self):
        ruleset = PermissionRuleset(default="ask")
        assert check_permission(ruleset, "write", "/any.txt") == "ask"

    def test_first_matching_rule_wins(self):
        ruleset = PermissionRuleset(
            read=OperationPermissions(
                default="allow",
                rules=[
                    PermissionRule("**/.env", "deny"),
                    PermissionRule("**/*", "allow"),
                ],
            ),
        )
        assert check_permission(ruleset, "read", "/project/.env") == "deny"

    def test_rule_order_matters(self):
        # Allow-first: the allow rule matches first
        ruleset = PermissionRuleset(
            read=OperationPermissions(
                default="deny",
                rules=[
                    PermissionRule("*.txt", "allow"),
                    PermissionRule("*.txt", "deny"),
                ],
            ),
        )
        assert check_permission(ruleset, "read", "file.txt") == "allow"

    def test_falls_through_to_default_when_no_match(self):
        ruleset = PermissionRuleset(
            write=OperationPermissions(
                default="ask",
                rules=[PermissionRule("*.log", "deny")],
            ),
        )
        assert check_permission(ruleset, "write", "/data.txt") == "ask"

    def test_get_operation_fallback(self):
        ruleset = PermissionRuleset(default="deny")
        op = ruleset.get_operation("execute")
        assert op.default == "deny"
        assert op.rules == []


# ---------------------------------------------------------------------------
# Preset validation
# ---------------------------------------------------------------------------


class TestReadonlyRuleset:
    def test_allows_read(self):
        assert check_permission(READONLY_RULESET, "read", "/file.txt") == "allow"

    def test_allows_glob(self):
        assert check_permission(READONLY_RULESET, "glob", "**/*.py") == "allow"

    def test_allows_grep(self):
        assert check_permission(READONLY_RULESET, "grep", "pattern") == "allow"

    def test_denies_write(self):
        assert check_permission(READONLY_RULESET, "write", "/file.txt") == "deny"

    def test_denies_edit(self):
        assert check_permission(READONLY_RULESET, "edit", "/file.txt") == "deny"

    def test_denies_execute(self):
        assert check_permission(READONLY_RULESET, "execute", "echo hi") == "deny"

    def test_denies_read_secrets(self):
        assert check_permission(READONLY_RULESET, "read", "/project/.env") == "deny"
        assert check_permission(READONLY_RULESET, "read", "/app/.env.local") == "deny"
        assert check_permission(READONLY_RULESET, "read", "/keys/server.pem") == "deny"


class TestDefaultRuleset:
    def test_allows_read(self):
        assert check_permission(DEFAULT_RULESET, "read", "/file.txt") == "allow"

    def test_asks_write(self):
        assert check_permission(DEFAULT_RULESET, "write", "/file.txt") == "ask"

    def test_asks_edit(self):
        assert check_permission(DEFAULT_RULESET, "edit", "/file.txt") == "ask"

    def test_asks_execute(self):
        assert check_permission(DEFAULT_RULESET, "execute", "echo hi") == "ask"

    def test_denies_read_secrets(self):
        assert check_permission(DEFAULT_RULESET, "read", "/project/.env") == "deny"

    def test_denies_write_secrets(self):
        assert check_permission(DEFAULT_RULESET, "write", "/project/.env") == "deny"

    def test_denies_dangerous_commands(self):
        assert check_permission(DEFAULT_RULESET, "execute", "rm -rf /") == "deny"


class TestPermissiveRuleset:
    def test_allows_read(self):
        assert check_permission(PERMISSIVE_RULESET, "read", "/file.txt") == "allow"

    def test_allows_write(self):
        assert check_permission(PERMISSIVE_RULESET, "write", "/file.txt") == "allow"

    def test_allows_execute(self):
        assert check_permission(PERMISSIVE_RULESET, "execute", "echo hi") == "allow"

    def test_denies_secrets(self):
        assert check_permission(PERMISSIVE_RULESET, "read", "/project/.env") == "deny"

    def test_denies_dangerous_commands(self):
        assert check_permission(PERMISSIVE_RULESET, "execute", "rm -rf /") == "deny"


class TestStrictRuleset:
    def test_asks_read(self):
        assert check_permission(STRICT_RULESET, "read", "/file.txt") == "ask"

    def test_asks_write(self):
        assert check_permission(STRICT_RULESET, "write", "/file.txt") == "ask"

    def test_asks_execute(self):
        assert check_permission(STRICT_RULESET, "execute", "echo hi") == "ask"

    def test_denies_secrets(self):
        assert check_permission(STRICT_RULESET, "read", "/project/.env") == "deny"

    def test_denies_dangerous_commands(self):
        assert check_permission(STRICT_RULESET, "execute", "rm -rf /") == "deny"
