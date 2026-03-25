"""Middleware protocol and implementations for langchain-agentkit.

The ``Middleware`` protocol defines the contract. Implementations live
in submodules: ``filesystem``, ``skills``, ``tasks``, ``hitl``, ``web_search``.

Re-exports::

    from langchain_agentkit.middleware import Middleware
    from langchain_agentkit.middleware import FilesystemMiddleware
    from langchain_agentkit.middleware import SkillsMiddleware
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from langchain_agentkit.middleware.filesystem import FilesystemMiddleware as FilesystemMiddleware
from langchain_agentkit.middleware.hitl import HITLMiddleware as HITLMiddleware
from langchain_agentkit.middleware.skills import SkillsMiddleware as SkillsMiddleware
from langchain_agentkit.middleware.tasks import TasksMiddleware as TasksMiddleware
from langchain_agentkit.middleware.web_search import (
    QwantSearchTool as QwantSearchTool,
)
from langchain_agentkit.middleware.web_search import (
    WebSearchMiddleware as WebSearchMiddleware,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime


class Middleware(Protocol):
    """Protocol for middleware that contributes tools and prompts to an agent."""

    @property
    def tools(self) -> list[BaseTool]:
        """Tools this middleware provides to the LLM."""
        ...

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime | None = None) -> str | None:
        """Prompt section to inject into the system prompt.

        Called on every LLM invocation. Return None to skip injection.
        """
        ...


__all__ = [
    "FilesystemMiddleware",
    "HITLMiddleware",
    "Middleware",
    "QwantSearchTool",
    "SkillsMiddleware",
    "TasksMiddleware",
    "WebSearchMiddleware",
]
