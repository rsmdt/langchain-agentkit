"""Tests for PromptComposition and AgentKit.compose()."""

from __future__ import annotations

import pytest

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.prompt_composition import PromptComposition


class _StrExt(Extension):
    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, state, runtime=None, *, tools=frozenset()):
        return self._text


class _DictExt(Extension):
    def __init__(self, *, prompt: str = "", reminder: str = "") -> None:
        self._prompt = prompt
        self._reminder = reminder

    def prompt(self, state, runtime=None, *, tools=frozenset()):
        out: dict[str, str] = {}
        if self._prompt:
            out["prompt"] = self._prompt
        if self._reminder:
            out["reminder"] = self._reminder
        return out


class _NoneExt(Extension):
    def prompt(self, state, runtime=None, *, tools=frozenset()):
        return None


class _EmptyStrExt(Extension):
    def prompt(self, state, runtime=None, *, tools=frozenset()):
        return ""


class TestPromptCompositionDataclass:
    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        comp = PromptComposition(prompt="a")
        with pytest.raises(FrozenInstanceError):
            comp.prompt = "x"  # type: ignore[misc]

    def test_default_is_empty(self):
        comp = PromptComposition()
        assert comp.prompt == ""
        assert comp.reminder == ""

    def test_prompt_and_reminder_channels(self):
        """PromptComposition carries two independent channels."""
        comp = PromptComposition(prompt="hello", reminder="<reminder>R</reminder>")
        assert comp.prompt == "hello"
        assert comp.reminder == "<reminder>R</reminder>"


class TestComposeReturnType:
    def test_returns_prompt_composition(self):
        kit = AgentKit(extensions=[])
        result = kit.compose({}, None)
        assert isinstance(result, PromptComposition)


class TestComposePromptAssembly:
    def test_empty_kit_returns_empty(self):
        kit = AgentKit(extensions=[])
        assert kit.compose({}, None).prompt == ""

    def test_base_prompt_only(self):
        kit = AgentKit(extensions=[], prompt="Base")
        assert kit.compose({}, None).prompt == "Base"

    def test_base_plus_extension_joined_with_double_newline(self):
        kit = AgentKit(extensions=[_StrExt("Section A")], prompt="Base")
        assert kit.compose({}, None).prompt == "Base\n\nSection A"

    def test_sections_joined_in_declaration_order(self):
        kit = AgentKit(
            extensions=[_StrExt("first"), _StrExt("second")],
            prompt="Base",
        )
        assert kit.compose({}, None).prompt == "Base\n\nfirst\n\nsecond"


class TestDictReturn:
    def test_dict_prompt_key_routed_to_system_prompt(self):
        kit = AgentKit(extensions=[_DictExt(prompt="P")])
        assert kit.compose({}, None).prompt == "P"

    def test_reminder_routed_to_reminder_channel_not_prompt(self):
        kit = AgentKit(extensions=[_DictExt(prompt="P", reminder="R")], prompt="Base")
        composition = kit.compose({}, None)
        # Durable content stays in the prompt channel; reminder is separate.
        assert composition.prompt == "Base\n\nP"
        assert "R" not in composition.prompt
        # Reminder is wrapped in a <reminder> envelope with a class subheader.
        assert composition.reminder.startswith("<reminder>")
        assert composition.reminder.endswith("</reminder>")
        assert "### _DictExt" in composition.reminder
        assert "R" in composition.reminder

    def test_reminder_without_prompt_contribution(self):
        kit = AgentKit(
            extensions=[_DictExt(reminder="only-reminder")],
            prompt="Base",
        )
        composition = kit.compose({}, None)
        # Base prompt unaffected; reminder lives in its own channel.
        assert composition.prompt == "Base"
        assert "only-reminder" in composition.reminder

    def test_multiple_reminders_collected_in_declaration_order(self):
        kit = AgentKit(
            extensions=[
                _DictExt(reminder="first-reminder"),
                _DictExt(reminder="second-reminder"),
            ],
        )
        reminder = kit.compose({}, None).reminder
        assert reminder.index("first-reminder") < reminder.index("second-reminder")
        # One envelope wraps all sections, not one per reminder.
        assert reminder.count("<reminder>") == 1
        assert reminder.count("</reminder>") == 1

    def test_reminder_empty_when_no_reminders(self):
        kit = AgentKit(extensions=[_DictExt(prompt="P")])
        composition = kit.compose({}, None)
        assert composition.reminder == ""
        assert "<reminder>" not in composition.prompt

    def test_dict_unknown_keys_ignored(self):
        class _WeirdExt(Extension):
            def prompt(self, state, runtime=None, *, tools=frozenset()):
                return {"prompt": "P", "other": "x"}

        kit = AgentKit(extensions=[_WeirdExt()])
        result = kit.compose({}, None).prompt
        assert "P" in result
        assert "other" not in result


class TestEmptyAndNoneReturns:
    def test_none_contributes_nothing(self):
        kit = AgentKit(extensions=[_NoneExt()], prompt="Base")
        assert kit.compose({}, None).prompt == "Base"

    def test_empty_string_contributes_nothing(self):
        kit = AgentKit(extensions=[_EmptyStrExt()], prompt="Base")
        assert kit.compose({}, None).prompt == "Base"
