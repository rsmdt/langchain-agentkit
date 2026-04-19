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


class TestAgentkitProtectionByDefault:
    """``.agentkit/`` is the agent's own configuration tree — every mutating
    preset must deny writes/edits there by default so self-modification does
    not leak through the bundled tools."""

    _MUTATING_PRESETS = [
        ("DEFAULT_RULESET", DEFAULT_RULESET),
        ("PERMISSIVE_RULESET", PERMISSIVE_RULESET),
        ("STRICT_RULESET", STRICT_RULESET),
    ]

    _AGENTKIT_PATHS = [
        ".agentkit/skills/deliverable/SKILL.md",
        "/.agentkit/skills/deliverable/SKILL.md",
        "/.agentkit/AGENTS.md",
        "/workspace/.agentkit/agents/interviewer.md",
        "/tmp/sandbox/.agentkit/prompts/system.md",
        ".agentkit",
        "/.agentkit",
    ]

    def test_all_mutating_presets_deny_write_under_agentkit(self):
        for preset_name, preset in self._MUTATING_PRESETS:
            for path in self._AGENTKIT_PATHS:
                assert check_permission(preset, "write", path) == "deny", (
                    f"{preset_name} should deny write {path!r}"
                )

    def test_all_mutating_presets_deny_edit_under_agentkit(self):
        for preset_name, preset in self._MUTATING_PRESETS:
            for path in self._AGENTKIT_PATHS:
                assert check_permission(preset, "edit", path) == "deny", (
                    f"{preset_name} should deny edit {path!r}"
                )

    def test_readonly_preset_still_denies_writes_generally(self):
        # READONLY denies all writes via default="deny" (no rules); verify the
        # agentkit paths still hit that default.
        for path in self._AGENTKIT_PATHS:
            assert check_permission(READONLY_RULESET, "write", path) == "deny"
            assert check_permission(READONLY_RULESET, "edit", path) == "deny"

    def test_agentkit_deny_wins_over_default_allow(self):
        # Sibling non-agentkit paths remain allowed in PERMISSIVE — proves
        # the new rule is targeted, not a blanket tightening.
        assert check_permission(PERMISSIVE_RULESET, "write", "/workspace/foo.md") == "allow"
        assert (
            check_permission(PERMISSIVE_RULESET, "write", "/workspace/.agentkit/foo.md") == "deny"
        )

    def test_reads_under_agentkit_still_allowed(self):
        # Seeding + SkillsExtension + AgentsExtension all read .agentkit/ —
        # protection is write-side only.
        assert check_permission(PERMISSIVE_RULESET, "read", "/.agentkit/skills/foo.md") == "allow"
        assert check_permission(DEFAULT_RULESET, "read", "/.agentkit/AGENTS.md") == "allow"

    def test_lookalike_paths_not_denied(self):
        # Regression guard: a future broadening of AGENTKIT_PATTERNS (e.g.,
        # to ``**/*.agentkit*``) must not silently pass. These paths share a
        # substring with ``.agentkit`` but are NOT inside the protected tree.
        for path in [
            "/workspace/.agentkitfoo/bar",
            "/workspace/my.agentkit.backup/x",
            "/workspace/agentkit/config.md",
            "/workspace/not-agentkit/file",
        ]:
            assert check_permission(PERMISSIVE_RULESET, "write", path) == "allow", (
                f"PERMISSIVE should allow write {path!r}"
            )
            assert check_permission(PERMISSIVE_RULESET, "edit", path) == "allow", (
                f"PERMISSIVE should allow edit {path!r}"
            )


class TestAgentkitShellProtection:
    """Shell commands that reference ``.agentkit`` are denied on every
    mutating preset's execute op — closes the Bash-bypass vector."""

    _MUTATING_PRESETS = [
        ("DEFAULT_RULESET", DEFAULT_RULESET),
        ("PERMISSIVE_RULESET", PERMISSIVE_RULESET),
        ("STRICT_RULESET", STRICT_RULESET),
    ]

    _AGENTKIT_COMMANDS = [
        "echo malicious > .agentkit/skills/payload.md",
        "tee .agentkit/AGENTS.md < /tmp/evil",
        "cp /tmp/evil.md ./.agentkit/skills/foo.md",
        "cat .agentkit/AGENTS.md",  # read via shell also denied — coarse by design
        "python3 -c \"open('.agentkit/agents/x.md','w').write('evil')\"",
    ]

    def test_all_mutating_presets_deny_agentkit_commands(self):
        for preset_name, preset in self._MUTATING_PRESETS:
            for command in self._AGENTKIT_COMMANDS:
                assert check_permission(preset, "execute", command) == "deny", (
                    f"{preset_name} should deny execute {command!r}"
                )

    def test_benign_commands_still_pass_preset_default(self):
        # PERMISSIVE passes benign commands through; DEFAULT/STRICT ask.
        # Regression guard that the pattern is targeted, not a blanket deny.
        assert check_permission(PERMISSIVE_RULESET, "execute", "ls /workspace") == "allow"
        assert check_permission(DEFAULT_RULESET, "execute", "ls /workspace") == "ask"
        assert check_permission(STRICT_RULESET, "execute", "ls /workspace") == "ask"

    def test_dangerous_commands_still_denied(self):
        # Baseline execute denies now contain both shell and dangerous rules —
        # verify DANGEROUS_COMMANDS coverage was not regressed by the merge.
        for preset_name, preset in self._MUTATING_PRESETS:
            assert check_permission(preset, "execute", "rm -rf /") == "deny", (
                f"{preset_name} lost DANGEROUS_COMMANDS coverage"
            )


class TestPermissionConstantsPublicApi:
    """Pattern constants are exported from the public permissions package
    so hosts composing custom rulesets can reuse them."""

    def test_agentkit_patterns_importable_from_package(self):
        from langchain_agentkit.permissions import AGENTKIT_PATTERNS

        assert AGENTKIT_PATTERNS
        assert "**/.agentkit/**" in AGENTKIT_PATTERNS

    def test_agentkit_command_patterns_importable_from_package(self):
        from langchain_agentkit.permissions import AGENTKIT_COMMAND_PATTERNS

        assert AGENTKIT_COMMAND_PATTERNS
        assert "**.agentkit**" in AGENTKIT_COMMAND_PATTERNS
