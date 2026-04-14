"""Cross-extension alignment checks.

Ensures each existing extension declares the expected ``prompt_cache_scope``
and, where applicable, that its ``prompt()`` output stays free of
tool-description-owned wording.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.extensions.web_search import WebSearchExtension


class TestFilesystemPromptAlignment:
    def test_no_tool_names_in_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = FilesystemExtension(root=tmpdir)
            result = ext.prompt({}) or ""
            for name in ("Read", "Write", "Edit", "Glob", "Grep", "Bash"):
                assert name not in result


class TestWebSearchPromptAlignment:
    def test_returns_none(self):
        ext = WebSearchExtension()
        assert ext.prompt({}, MagicMock()) is None
