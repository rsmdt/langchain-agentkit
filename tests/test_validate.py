"""Tests for skill config validation (AgentSkills.io compliance)."""

from pathlib import Path

from langchain_agentkit.types import SkillConfig
from langchain_agentkit.validate import validate_skill_config


class TestValidateSkillConfig:
    def test_valid_config_returns_no_errors(self):
        config = SkillConfig(
            name="market-sizing",
            description="Size markets",
            directory=Path("market-sizing"),
        )

        errors = validate_skill_config(config)

        assert errors == []

    def test_missing_name_returns_error(self):
        config = SkillConfig(name="", description="Some description")

        errors = validate_skill_config(config)

        assert any("name" in e for e in errors)

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


class TestDirectoryNameMatch:
    def test_name_must_match_directory(self):
        config = SkillConfig(
            name="market-sizing",
            description="desc",
            directory=Path("wrong-name"),
        )

        errors = validate_skill_config(config)

        assert any("does not match directory" in e for e in errors)

    def test_matching_directory_passes(self):
        config = SkillConfig(
            name="market-sizing",
            description="desc",
            directory=Path("market-sizing"),
        )

        errors = validate_skill_config(config)

        assert not any("does not match directory" in e for e in errors)

    def test_skips_directory_check_for_default(self):
        config = SkillConfig(name="market-sizing", description="desc")

        errors = validate_skill_config(config)

        assert not any("does not match directory" in e for e in errors)
