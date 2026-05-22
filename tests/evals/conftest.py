"""Shared fixtures for eval tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langchain_agentkit.extensions.skills import SkillsExtension
from langchain_agentkit.extensions.skills.types import SkillConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Load .env file at project root (no third-party dependency needed)
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Single source of truth for the eval model. Override per-run with the
# AGENTKIT_EVAL_MODEL env var; before/after baselines MUST use the same
# model or the comparison is meaningless. Read after .env load above so a
# project-local .env can set it. pytest imports conftest before any test
# module, so importers see the resolved value.
EVAL_MODEL = os.environ.get("AGENTKIT_EVAL_MODEL", "gpt-5.4-mini")


def _load_skills_from_fixtures() -> list[SkillConfig]:
    """Load SkillConfig objects from test fixture directories."""
    from langchain_agentkit.frontmatter import parse_frontmatter

    configs: list[SkillConfig] = []
    skills_dir = FIXTURES / "skills"
    if skills_dir.exists():
        for d in skills_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                result = parse_frontmatter(d / "SKILL.md")
                configs.append(SkillConfig.from_frontmatter(result.metadata, result.content))
    return configs


@pytest.fixture()
def skills_extension() -> SkillsExtension:
    """SkillsExtension wired to test fixtures."""
    return SkillsExtension(skills=_load_skills_from_fixtures())
