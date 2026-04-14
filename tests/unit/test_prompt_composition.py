"""Tests for PromptComposition and AgentKit.compose()."""

from __future__ import annotations

import pytest

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.prompt_composition import PromptComposition


class _StrExt(Extension):
    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, state, runtime=None):
        return self._text


class _DictExt(Extension):
    def __init__(self, *, prompt: str = "", reminder: str = "") -> None:
        self._prompt = prompt
        self._reminder = reminder

    def prompt(self, state, runtime=None):
        out: dict[str, str] = {}
        if self._prompt:
            out["prompt"] = self._prompt
        if self._reminder:
            out["reminder"] = self._reminder
        return out


class _NoneExt(Extension):
    def prompt(self, state, runtime=None):
        return None


class _EmptyStrExt(Extension):
    def prompt(self, state, runtime=None):
        return ""


class TestPromptCompositionDataclass:
    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        comp = PromptComposition(prompt="a", reminder="b")
        with pytest.raises(FrozenInstanceError):
            comp.prompt = "x"  # type: ignore[misc]

    def test_default_fields_are_empty(self):
        comp = PromptComposition()
        assert comp.prompt == ""
        assert comp.reminder == ""


class TestComposeReturnType:
    def test_returns_prompt_composition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[])
        result = kit.compose({}, None)
        assert isinstance(result, PromptComposition)


class TestComposePromptJoining:
    def test_base_prompt_only(self):
        kit = AgentKit(extensions=[], prompt="Base prompt")
        result = kit.compose({}, None)
        assert result.prompt == "Base prompt"

    def test_empty_base_prompt(self):
        kit = AgentKit(extensions=[])
        result = kit.compose({}, None)
        assert result.prompt == ""

    def test_base_plus_extension_joined_with_double_newline(self):
        kit = AgentKit(extensions=[_StrExt("Section A")], prompt="Base")
        result = kit.compose({}, None)
        assert result.prompt == "Base\n\nSection A"

    def test_sections_joined_in_declaration_order(self):
        kit = AgentKit(
            extensions=[_StrExt("first"), _StrExt("second")],
            prompt="Base",
        )
        result = kit.compose({}, None)
        assert result.prompt == "Base\n\nfirst\n\nsecond"


class TestDictReturn:
    def test_dict_prompt_routed_to_prompt_channel(self):
        kit = AgentKit(extensions=[_DictExt(prompt="P")])
        result = kit.compose({}, None)
        assert result.prompt == "P"

    def test_dict_reminder_appended_to_reminder_channel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[_DictExt(reminder="R")])
        result = kit.compose({}, None)
        assert "# _DictExt" in result.reminder
        assert "R" in result.reminder

    def test_dict_unknown_keys_ignored(self):
        class _WeirdExt(Extension):
            def prompt(self, state, runtime=None):
                return {"prompt": "P", "other": "x", "static": "ignored", "dynamic": "ignored"}

        kit = AgentKit(extensions=[_WeirdExt()])
        result = kit.compose({}, None)
        assert result.prompt == "P"
        assert "ignored" not in result.prompt
        assert "ignored" not in result.reminder


class TestEmptyAndNoneReturns:
    def test_none_contributes_nothing(self):
        kit = AgentKit(extensions=[_NoneExt()], prompt="Base")
        result = kit.compose({}, None)
        assert result.prompt == "Base"

    def test_empty_string_contributes_nothing(self):
        kit = AgentKit(extensions=[_EmptyStrExt()], prompt="Base")
        result = kit.compose({}, None)
        assert result.prompt == "Base"
