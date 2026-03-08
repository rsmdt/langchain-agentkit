"""Middleware protocol for langchain-agentkit.

A Middleware contributes two things to an agent:

- **tools**: LangChain ``BaseTool`` instances the LLM can call.
- **prompt**: A string section injected into the system prompt on every
  LLM invocation, or ``None`` to skip injection for that call.

Implement this protocol to create reusable agent capabilities that can be
composed together via ``AgentKit``. Each middleware is independent — it
manages its own tools and prompt logic — so concerns stay separate and
individual middlewares remain unit-testable in isolation.

Example::

    class MyMiddleware:
        @property
        def tools(self) -> list[BaseTool]:
            return [my_tool]

        def prompt(self, state: dict, runtime: ToolRuntime) -> str | None:
            return "You have access to my_tool."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolRuntime


class Middleware(Protocol):
    @property
    def tools(self) -> list[BaseTool]:
        """Tools this middleware provides to the LLM."""
        ...

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime) -> str | None:
        """Prompt section to inject into the system prompt.

        Called on every LLM invocation. Return None to skip injection.

        Args:
            state: Current graph state.
            runtime: Tool runtime context. Use ``runtime.config`` for
                the full ``RunnableConfig``.
        """
        ...
