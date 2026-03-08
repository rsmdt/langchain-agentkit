"""ToolRuntime — unified runtime context for tools and middleware.

LangGraph auto-injects ``ToolRuntime`` into tool signatures just like
``RunnableConfig``. The parameter is hidden from the LLM schema.

Access the full ``RunnableConfig`` via ``runtime.config`` when needed.

Example — middleware prompt::

    def prompt(self, state: dict, runtime: ToolRuntime) -> str | None:
        thread_id = runtime.config.get("configurable", {}).get("thread_id")
        return f"Thread: {thread_id}"

Example — agent handler::

    async def handler(state, *, llm, runtime):
        config = runtime.config
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class ToolRuntime:
    """Unified runtime context for tools and middleware.

    Wraps LangGraph's ``RunnableConfig`` and any additional runtime
    kwargs into a single injectable parameter. Hidden from the LLM
    schema — only visible to tool/middleware implementations.

    Attributes:
        config: The full ``RunnableConfig`` for the current invocation.
    """

    __slots__ = ("_config", "_extras")

    def __init__(self, config: RunnableConfig, **extras: Any) -> None:
        self._config = config
        self._extras = extras

    @property
    def config(self) -> RunnableConfig:
        """The full ``RunnableConfig`` for the current invocation."""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get an extra runtime value by key.

        Allows forward-compatible access to additional runtime context
        without breaking existing signatures.
        """
        return self._extras.get(key, default)

    def __repr__(self) -> str:
        extras = f", extras={set(self._extras)}" if self._extras else ""
        return f"ToolRuntime(config=...{extras})"
