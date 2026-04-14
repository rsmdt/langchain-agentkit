"""Tests for CoreBehaviorExtension."""

from __future__ import annotations

import platform
from pathlib import Path

from langchain_agentkit.extensions.core_behavior import CoreBehaviorExtension


class TestCoreBehaviorExtensionConstruction:
    def test_no_filesystem_io_in_init(self, monkeypatch):
        """Constructor must not touch the filesystem or environment."""
        called = {"n": 0}

        def boom() -> Path:
            called["n"] += 1
            return Path("/tmp")

        CoreBehaviorExtension(cwd=boom)
        assert called["n"] == 0

    def test_defaults(self):
        ext = CoreBehaviorExtension()
        assert ext.tools == []
        assert ext.state_schema is None

    def test_prompt_cache_scope_is_static(self):
        assert CoreBehaviorExtension.prompt_cache_scope == "static"


class TestPromptBody:
    def test_prompt_returns_string(self):
        ext = CoreBehaviorExtension(include_env=False)
        out = ext.prompt({}, None)
        assert isinstance(out, str)
        assert out.strip() != ""

    def test_body_is_domain_neutral(self):
        ext = CoreBehaviorExtension(include_env=False)
        body = ext.prompt({}, None).lower()
        for forbidden in ("code", "test", "commit", "pull request", " pr ", "software"):
            assert forbidden not in body, f"found forbidden token: {forbidden!r}"

    def test_body_under_2kb(self):
        ext = CoreBehaviorExtension(include_env=False)
        body = ext.prompt({}, None)
        assert len(body.encode("utf-8")) <= 2048


class TestPromptEnvBlock:
    def test_env_included_by_default(self, tmp_path):
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "<env>" in out and "</env>" in out
        assert "Primary working directory:" in out
        assert "Is a git repository:" in out
        assert "Platform:" in out
        assert "Shell:" in out
        assert "OS Version:" in out

    def test_env_excluded_when_include_env_false(self):
        ext = CoreBehaviorExtension(include_env=False)
        out = ext.prompt({}, None)
        assert "<env>" not in out

    def test_cwd_callable_invoked_every_turn(self, tmp_path):
        calls = {"n": 0}

        def cwd() -> Path:
            calls["n"] += 1
            return tmp_path

        ext = CoreBehaviorExtension(cwd=cwd)
        ext.prompt({}, None)
        ext.prompt({}, None)
        assert calls["n"] == 2

    def test_cwd_reports_callable_value(self, tmp_path):
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert f"Primary working directory: {tmp_path}" in out

    def test_cwd_default_uses_path_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        ext = CoreBehaviorExtension()
        out = ext.prompt({}, None)
        assert f"Primary working directory: {tmp_path}" in out

    def test_is_git_false_for_non_repo(self, tmp_path):
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Is a git repository: False" in out

    def test_is_git_true_for_standard_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        ext = CoreBehaviorExtension(cwd=lambda: sub)
        out = ext.prompt({}, None)
        assert "Is a git repository: True" in out

    def test_is_git_true_for_submodule(self, tmp_path):
        (tmp_path / ".git").write_text("gitdir: ../.git/modules/sub\n")
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Is a git repository: True" in out

    def test_worktree_line_present_when_git_is_file(self, tmp_path):
        (tmp_path / ".git").write_text("gitdir: /some/where/.git/worktrees/x\n")
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "worktree" in out
        assert "do NOT cd to original repo" in out

    def test_worktree_line_absent_for_standard_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "do NOT cd to original repo" not in out

    def test_platform_and_os_version_auto_detected(self, tmp_path):
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert f"Platform: {platform.system()} {platform.release()}" in out
        assert f"OS Version: {platform.platform()}" in out

    def test_shell_from_environ(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Shell: zsh" in out

    def test_shell_missing_renders_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHELL", raising=False)
        ext = CoreBehaviorExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Shell: \n" in out or "Shell:\n" in out or out.rstrip().endswith("Shell:")

    def test_no_model_family_or_cutoff_strings(self, tmp_path):
        neutral = tmp_path.parent / "neutral"
        neutral.mkdir(exist_ok=True)
        ext = CoreBehaviorExtension(cwd=lambda: neutral)
        out = ext.prompt({}, None)
        lines = [
            line for line in out.splitlines() if not line.startswith("Primary working directory:")
        ]
        text = "\n".join(lines).lower()
        for forbidden in ("claude", "gpt", "anthropic", "openai", "knowledge cutoff", "cutoff"):
            assert forbidden not in text


class TestDependencies:
    def test_no_dependencies(self):
        ext = CoreBehaviorExtension()
        assert ext.dependencies() == []
