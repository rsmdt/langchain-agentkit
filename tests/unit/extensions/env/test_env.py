"""Tests for EnvExtension."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

from langchain_agentkit.extensions.env import EnvExtension

if TYPE_CHECKING:
    from pathlib import Path


class TestPromptEnvBlock:
    def test_env_included_by_default(self, tmp_path):
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "<env>" in out and "</env>" in out
        assert "Primary working directory:" in out
        assert "Is a git repository:" in out
        assert "Platform:" in out
        assert "Shell:" in out
        assert "OS Version:" in out

    def test_cwd_callable_invoked_every_turn(self, tmp_path):
        calls = {"n": 0}

        def cwd() -> Path:
            calls["n"] += 1
            return tmp_path

        ext = EnvExtension(cwd=cwd)
        ext.prompt({}, None)
        ext.prompt({}, None)
        assert calls["n"] == 2

    def test_cwd_reports_callable_value(self, tmp_path):
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert f"Primary working directory: {tmp_path}" in out

    def test_cwd_default_uses_path_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        ext = EnvExtension()
        out = ext.prompt({}, None)
        assert f"Primary working directory: {tmp_path}" in out

    def test_is_git_false_for_non_repo(self, tmp_path):
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Is a git repository: False" in out

    def test_is_git_true_for_standard_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        ext = EnvExtension(cwd=lambda: sub)
        out = ext.prompt({}, None)
        assert "Is a git repository: True" in out

    def test_is_git_true_for_submodule(self, tmp_path):
        (tmp_path / ".git").write_text("gitdir: ../.git/modules/sub\n")
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Is a git repository: True" in out

    def test_worktree_line_present_when_git_is_file(self, tmp_path):
        (tmp_path / ".git").write_text("gitdir: /some/where/.git/worktrees/x\n")
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "worktree" in out
        assert "do NOT cd to original repo" in out

    def test_worktree_line_absent_for_standard_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "do NOT cd to original repo" not in out

    def test_platform_and_os_version_auto_detected(self, tmp_path):
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert f"Platform: {platform.system()} {platform.release()}" in out
        assert f"OS Version: {platform.platform()}" in out

    def test_shell_from_environ(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHELL", "/usr/bin/zsh")
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Shell: zsh" in out

    def test_shell_missing_renders_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHELL", raising=False)
        ext = EnvExtension(cwd=lambda: tmp_path)
        out = ext.prompt({}, None)
        assert "Shell: \n" in out or "Shell:\n" in out or out.rstrip().endswith("Shell:")

    def test_no_model_family_or_cutoff_strings(self, tmp_path):
        neutral = tmp_path.parent / "neutral"
        neutral.mkdir(exist_ok=True)
        ext = EnvExtension(cwd=lambda: neutral)
        out = ext.prompt({}, None)
        lines = [
            line for line in out.splitlines() if not line.startswith("Primary working directory:")
        ]
        text = "\n".join(lines).lower()
        for forbidden in ("claude", "gpt", "anthropic", "openai", "knowledge cutoff", "cutoff"):
            assert forbidden not in text
