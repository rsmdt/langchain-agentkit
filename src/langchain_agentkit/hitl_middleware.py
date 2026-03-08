"""HITLMiddleware — human-in-the-loop tool call approval via LangGraph interrupt().

Wraps individual tool calls for human review before execution. Uses
LangGraph's ``interrupt()`` to pause the graph and ``Command(resume=...)``
to continue with the human's decision.

Usage::

    from langchain_agentkit import HITLMiddleware

    mw = HITLMiddleware(interrupt_on={
        "send_email": True,                          # all decisions
        "delete_file": {"allowed_decisions": ["approve", "reject"]},
        "search": False,                             # auto-approved (excluded)
    })

    class my_agent(agent):
        llm = ChatOpenAI(model="gpt-4o")
        tools = [send_email, delete_file, search]
        middleware = [mw]

        async def handler(state, *, llm, tools, prompt, config, runtime):
            ...

    # Compile with checkpointer (required for interrupt)
    graph = my_agent.compile(checkpointer=InMemorySaver())

    # Invoke — pauses at interrupt, resume with Command(resume=...)
    result = graph.invoke(input, config)
    graph.invoke(Command(resume={"type": "approve"}), config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.types import interrupt

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

    from langchain_agentkit.runtime import ToolRuntime

DecisionType = Literal["approve", "edit", "reject"]


class InterruptConfig:
    """Configuration for a tool requiring human approval.

    Args:
        allowed_decisions: Which decisions the human can make.
        description: Static string or callable that generates a description
            for the interrupt request. Callable receives ``(tool_call, state)``.
    """

    __slots__ = ("allowed_decisions", "description")

    def __init__(
        self,
        allowed_decisions: list[DecisionType],
        description: str | Callable[..., str] | None = None,
    ) -> None:
        self.allowed_decisions = allowed_decisions
        self.description = description


class HITLMiddleware:
    """Middleware providing human-in-the-loop tool call approval.

    Uses LangGraph's ``interrupt()`` inside a ``wrap_tool_call`` callback
    to pause execution before configured tools run. The human reviews the
    tool call and responds with approve, edit, or reject.

    This middleware satisfies the ``Middleware`` protocol (``tools`` +
    ``prompt``) and additionally provides ``wrap_tool_call`` which the
    ``agent`` metaclass passes to ``ToolNode``.

    Args:
        interrupt_on: Mapping of tool name to approval config.

            - ``True``: all decisions allowed (approve, edit, reject)
            - ``False``: auto-approved (excluded from HITL)
            - ``dict``: explicit config with ``allowed_decisions`` key
            - ``InterruptConfig``: full config object

    Example::

        mw = HITLMiddleware(interrupt_on={
            "send_email": True,
            "search": False,
            "delete_file": {"allowed_decisions": ["approve", "reject"]},
        })
    """

    def __init__(
        self,
        interrupt_on: dict[str, bool | dict[str, Any] | InterruptConfig],
    ) -> None:
        resolved: dict[str, InterruptConfig] = {}
        for tool_name, config in interrupt_on.items():
            if isinstance(config, bool):
                if config:
                    resolved[tool_name] = InterruptConfig(
                        allowed_decisions=["approve", "edit", "reject"]
                    )
            elif isinstance(config, InterruptConfig):
                resolved[tool_name] = config
            elif isinstance(config, dict) and config.get("allowed_decisions"):
                resolved[tool_name] = InterruptConfig(
                    allowed_decisions=config["allowed_decisions"],
                    description=config.get("description"),
                )
        self.interrupt_on = resolved

    @property
    def tools(self) -> list[BaseTool]:
        """No tools — HITL is a tool execution wrapper, not a tool provider."""
        return []

    def prompt(self, state: dict[str, Any], runtime: ToolRuntime) -> str | None:
        """No prompt injection needed."""
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: Callable[[ToolCallRequest], ToolMessage | Command],  # type: ignore[type-arg]
    ) -> ToolMessage | Command:  # type: ignore[type-arg]
        """Intercept tool calls for human approval before execution.

        Auto-approved tools pass straight through. Configured tools
        trigger an ``interrupt()`` that pauses the graph until the human
        responds via ``Command(resume=...)``.

        Resume payload::

            {"type": "approve"}
            {"type": "edit", "args": {"to": "new@example.com"}}
            {"type": "reject", "message": "Not now"}

        Args:
            request: The tool call request from ToolNode.
            execute: Callback to execute the tool call.
        """
        tool_name = request.tool_call["name"]
        config = self.interrupt_on.get(tool_name)

        if config is None:
            return execute(request)

        description = self._build_description(request, config)

        decision = interrupt(
            {
                "tool": tool_name,
                "args": request.tool_call["args"],
                "allowed_decisions": config.allowed_decisions,
                "description": description,
            }
        )

        decision_type = decision.get("type", "reject") if isinstance(decision, dict) else "reject"

        if decision_type == "approve" and "approve" in config.allowed_decisions:
            return execute(request)

        if decision_type == "edit" and "edit" in config.allowed_decisions:
            edited_args = decision.get("args", request.tool_call["args"])
            modified_call = ToolCall(
                name=request.tool_call["name"],
                args=edited_args,
                id=request.tool_call["id"],
                type="tool_call",
            )
            return execute(request.override(tool_call=modified_call))

        if decision_type == "reject" and "reject" in config.allowed_decisions:
            message = decision.get("message", f"User rejected {tool_name}")
            return ToolMessage(
                content=message,
                name=tool_name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        return ToolMessage(
            content=f"Invalid decision '{decision_type}' for {tool_name}. "
            f"Allowed: {config.allowed_decisions}",
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def _build_description(
        self,
        request: ToolCallRequest,
        config: InterruptConfig,
    ) -> str:
        """Build a human-readable description for the interrupt request."""
        if config.description is None:
            tool_name = request.tool_call["name"]
            tool_args = request.tool_call["args"]
            return f"Tool: {tool_name}\nArgs: {tool_args}"
        if callable(config.description):
            return config.description(request.tool_call)
        return config.description
