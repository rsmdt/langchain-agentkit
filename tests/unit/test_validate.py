"""Tests for skill config validation (AgentSkills.io compliance)."""

from langchain_agentkit.extensions.skills.types import SkillConfig
from langchain_agentkit.extensions.skills.discovery import validate_skill_config


class TestValidateSkillConfig:
    def test_valid_config_returns_no_errors(self):
        config = SkillConfig(name="market-sizing", description="Size markets")

        errors = validate_skill_config(config)

        assert errors == []

    def test_missing_name_returns_error(self):
        config = SkillConfig(name="", description="Some description")

        errors = validate_skill_config(config)

        assert any("name" in e.lower() for e in errors)

    def test_missing_description_returns_error(self):
        config = SkillConfig(name="test-skill", description="")

        errors = validate_skill_config(config)

        assert any("description" in e for e in errors)

    def test_missing_both_returns_two_errors(self):
        config = SkillConfig(name="", description="")

        errors = validate_skill_config(config)

        assert len(errors) == 2


class TestNameFormat:
    def test_rejects_uppercase(self):
        config = SkillConfig(name="Market-Sizing", description="desc")

        errors = validate_skill_config(config)

        assert any("invalid" in e.lower() for e in errors)

    def test_rejects_underscores(self):
        config = SkillConfig(name="market_sizing", description="desc")

        errors = validate_skill_config(config)

        assert any("invalid" in e.lower() for e in errors)

    def test_rejects_leading_hyphen(self):
        config = SkillConfig(name="-market", description="desc")

        errors = validate_skill_config(config)

        assert any("invalid" in e.lower() for e in errors)

    def test_rejects_trailing_hyphen(self):
        config = SkillConfig(name="market-", description="desc")

        errors = validate_skill_config(config)

        assert any("invalid" in e.lower() for e in errors)

    def test_rejects_consecutive_hyphens(self):
        config = SkillConfig(name="market--sizing", description="desc")

        errors = validate_skill_config(config)

        assert any("invalid" in e.lower() for e in errors)

    def test_accepts_single_char(self):
        config = SkillConfig(name="a", description="desc")

        errors = validate_skill_config(config)

        assert not any("invalid" in e.lower() for e in errors)

    def test_accepts_digits(self):
        config = SkillConfig(name="skill2", description="desc")

        errors = validate_skill_config(config)

        assert not any("invalid" in e.lower() for e in errors)
