"""Tests for PromptComposition and AgentKit.compose()."""

from __future__ import annotations

import pytest

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extension import Extension
from langchain_agentkit.prompt_composition import PromptComposition


class _StaticExt(Extension):
    prompt_cache_scope = "static"

    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, state, runtime=None):
        return self._text


class _DynamicExt(Extension):
    # default prompt_cache_scope is "dynamic"
    def __init__(self, text: str) -> None:
        self._text = text

    def prompt(self, state, runtime=None):
        return self._text


class _StaticDictExt(Extension):
    prompt_cache_scope = "static"

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


class _DynamicDictExt(Extension):
    # default scope = dynamic
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
    prompt_cache_scope = "static"

    def prompt(self, state, runtime=None):
        return ""


class TestPromptCompositionDataclass:
    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        comp = PromptComposition(static="a", dynamic="b", reminder="c")
        with pytest.raises(FrozenInstanceError):
            comp.static = "x"  # type: ignore[misc]

    def test_default_fields_are_empty(self):
        comp = PromptComposition()
        assert comp.static == ""
        assert comp.dynamic == ""
        assert comp.reminder == ""

    def test_joined_concatenates_with_double_newline(self):
        comp = PromptComposition(static="A", dynamic="B")
        assert comp.joined == "A\n\nB"

    def test_joined_filters_empty_static(self):
        comp = PromptComposition(static="", dynamic="B")
        assert comp.joined == "B"

    def test_joined_filters_empty_dynamic(self):
        comp = PromptComposition(static="A", dynamic="")
        assert comp.joined == "A"

    def test_joined_all_empty(self):
        comp = PromptComposition()
        assert comp.joined == ""

    def test_joined_excludes_reminder(self):
        comp = PromptComposition(static="A", dynamic="B", reminder="R")
        assert "R" not in comp.joined


class TestComposeReturnType:
    def test_returns_prompt_composition(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[])
        result = kit.compose({}, None)
        assert isinstance(result, PromptComposition)


class TestBasePromptRoutesToStatic:
    def test_base_prompt_goes_to_static(self):
        kit = AgentKit(extensions=[], prompt="Base prompt")
        result = kit.compose({}, None)
        assert result.static == "Base prompt"
        assert result.dynamic == ""

    def test_empty_base_prompt_no_static(self):
        kit = AgentKit(extensions=[])
        result = kit.compose({}, None)
        assert result.static == ""


class TestPromptCacheScope:
    def test_default_scope_is_dynamic(self):
        assert Extension.prompt_cache_scope == "dynamic"

    def test_str_return_routed_to_static_when_declared(self):
        kit = AgentKit(extensions=[_StaticExt("stable")])
        result = kit.compose({}, None)
        assert result.static == "stable"
        assert result.dynamic == ""

    def test_str_return_routed_to_dynamic_by_default(self):
        kit = AgentKit(extensions=[_DynamicExt("live")])
        result = kit.compose({}, None)
        assert result.static == ""
        assert result.dynamic == "live"

    def test_base_prompt_and_static_extension_combined(self):
        kit = AgentKit(extensions=[_StaticExt("stable")], prompt="Base")
        result = kit.compose({}, None)
        assert result.static == "Base\n\nstable"


class TestDictReturn:
    def test_dict_prompt_key_routed_by_scope_static(self):
        kit = AgentKit(extensions=[_StaticDictExt(prompt="S")])
        result = kit.compose({}, None)
        assert result.static == "S"
        assert result.dynamic == ""

    def test_dict_prompt_key_routed_by_scope_dynamic(self):
        kit = AgentKit(extensions=[_DynamicDictExt(prompt="D")])
        result = kit.compose({}, None)
        assert result.static == ""
        assert result.dynamic == "D"

    def test_dict_reminder_appended_to_reminder_channel(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        kit = AgentKit(extensions=[_DynamicDictExt(reminder="R")])
        result = kit.compose({}, None)
        assert "R" in result.reminder

    def test_dict_unknown_keys_ignored(self):
        class _WeirdExt(Extension):
            prompt_cache_scope = "static"

            def prompt(self, state, runtime=None):
                return {"prompt": "S", "other": "x", "static": "ignored", "dynamic": "ignored"}

        kit = AgentKit(extensions=[_WeirdExt()])
        result = kit.compose({}, None)
        assert result.static == "S"
        assert result.dynamic == ""
        assert "ignored" not in result.static
        assert "ignored" not in result.dynamic

    def test_dict_static_dynamic_keys_no_longer_recognized(self):
        """The legacy ``{static, dynamic}`` dict shape is ignored now."""

        class _LegacyExt(Extension):
            prompt_cache_scope = "static"

            def prompt(self, state, runtime=None):
                return {"static": "legacy-static", "dynamic": "legacy-dynamic"}

        kit = AgentKit(extensions=[_LegacyExt()])
        result = kit.compose({}, None)
        assert "legacy-static" not in result.static
        assert "legacy-dynamic" not in result.dynamic


class TestEmptyAndNoneReturns:
    def test_none_contributes_nothing(self):
        kit = AgentKit(extensions=[_NoneExt()], prompt="Base")
        result = kit.compose({}, None)
        assert result.static == "Base"
        assert result.dynamic == ""

    def test_empty_string_contributes_nothing(self):
        kit = AgentKit(extensions=[_EmptyStrExt()], prompt="Base")
        result = kit.compose({}, None)
        assert result.static == "Base"


class TestSectionOrdering:
    def test_sections_joined_in_declaration_order(self):
        kit = AgentKit(
            extensions=[_StaticExt("first"), _StaticExt("second")],
            prompt="Base",
        )
        result = kit.compose({}, None)
        assert result.static == "Base\n\nfirst\n\nsecond"

    def test_dynamic_sections_joined_in_order(self):
        kit = AgentKit(
            extensions=[_DynamicExt("one"), _DynamicExt("two")],
        )
        result = kit.compose({}, None)
        assert result.dynamic == "one\n\ntwo"

    def test_mixed_scopes_preserved(self):
        kit = AgentKit(
            extensions=[_DynamicExt("D1"), _StaticExt("S1"), _DynamicExt("D2")],
            prompt="Base",
        )
        result = kit.compose({}, None)
        assert result.static == "Base\n\nS1"
        assert result.dynamic == "D1\n\nD2"
