"""Tests for AgentKit's built-in reminder channel."""

from __future__ import annotations

import datetime as _dt

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension

REFERENCE_DISCLAIMER = (
    "IMPORTANT: this context may or may not be relevant to your tasks. "
    "You should not respond to this context unless it is highly relevant to your task."
)


class TestBuiltinDate:
    def test_date_always_included(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[])
        reminder = kit.compose({}, None).reminder
        today = _dt.date.today().strftime("%Y-%m-%d")
        assert reminder.startswith("<system-reminder>")
        assert reminder.endswith("</system-reminder>")
        assert "As you answer the user's questions, you can use the following context:" in reminder
        assert "# currentDate" in reminder
        assert f"Today's date is {today}." in reminder
        assert REFERENCE_DISCLAIMER in reminder

    def test_agents_md_not_autodiscovered(self, tmp_path, monkeypatch):
        """AGENTS.md files in cwd are no longer auto-injected."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("project rules")
        kit = AgentKit(extensions=[])
        reminder = kit.compose({}, None).reminder
        assert "# agentsMd" not in reminder
        assert "project rules" not in reminder


class TestExtensionReminderContribution:
    def test_extension_reminder_appended(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        class _Ext(Extension):
            def prompt(self, state, runtime=None):
                return {"reminder": "hello world"}

        kit = AgentKit(extensions=[_Ext()])
        reminder = kit.compose({}, None).reminder
        assert "# _Ext" in reminder
        assert "hello world" in reminder

    def test_extension_reminder_plus_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        class _Ext(Extension):
            def prompt(self, state, runtime=None):
                return {"prompt": "S", "reminder": "R"}

        kit = AgentKit(extensions=[_Ext()])
        result = kit.compose({}, None)
        assert "S" in result.prompt
        assert "R" in result.reminder

    def test_empty_reminder_value_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        class _Ext(Extension):
            def prompt(self, state, runtime=None):
                return {"reminder": ""}

        kit = AgentKit(extensions=[_Ext()])
        reminder = kit.compose({}, None).reminder
        # Date section still present, but no _Ext section.
        assert "# _Ext" not in reminder


class TestOutputTemplate:
    def test_header_and_disclaimer_envelope(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[])
        reminder = kit.compose({}, None).reminder
        lines = reminder.splitlines()
        assert lines[0] == "<system-reminder>"
        assert lines[1] == "As you answer the user's questions, you can use the following context:"
        assert lines[-1] == "</system-reminder>"
        assert lines[-2] == REFERENCE_DISCLAIMER
        assert "# currentDate" in reminder
