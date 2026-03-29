"""Shared fixtures for eval tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from langchain_agentkit.extensions.skills import SkillsExtension

if TYPE_CHECKING:
    from langchain_agentkit.vfs import VirtualFilesystem

FIXTURES = Path(__file__).parent.parent / "fixtures"

# Load .env file at project root (no third-party dependency needed)
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture()
def skills_extension() -> SkillsExtension:
    """SkillsExtension wired to test fixtures."""
    return SkillsExtension(str(FIXTURES / "skills"))


@pytest.fixture()
def vfs_with_workspace(skills_extension: SkillsExtension) -> VirtualFilesystem:
    """VFS with skills loaded + workspace files for edit/write tests."""
    vfs = skills_extension.filesystem
    vfs.write("/workspace/config.json", '{"debug": true, "verbose": false}')
    vfs.write("/workspace/notes.txt", "Some notes here")
    return vfs
