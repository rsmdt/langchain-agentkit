"""HITLExtension — human-in-the-loop via unified Question protocol.

Provides two capabilities:

1. **Tool approval**: Intercepts whitelisted tool calls via a
   ``wrap_tool`` hook and presents structured questions
   (Approve/Edit/Reject) before execution.
2. **ask_user tool**: Gives the LLM an explicit tool to ask the user
   structured questions during execution.

Both use the same Question-based interrupt protocol. Consumers receive
a unified payload format regardless of the interrupt source.

Usage::

    from langchain_agentkit import HITLExtension

    # Tool approval only
    hitl = HITLExtension(interrupt_on={"send_email": True})

    # ask_user tool only
    hitl = HITLExtension(tools=True)

    # Both
    hitl = HITLExtension(
        interrupt_on={"send_email": True, "delete_file": True},
        tools=True,
    )

    # Custom approval config
    hitl = HITLExtension(interrupt_on={
        "send_email": {"options": ["approve", "reject"], "question": "Send email?"},
    })

Interrupt payload (unified for both tool approval and ask_user)::

    {
        "type": "question",
        "questions": [
            {
                "question": "Send email?",
                "header": "send_email",
                "options": [
                    {"label": "Approve", "description": "Execute as-is"},
                    {"label": "Reject", "description": "Deny this call"}
                ],
                "multi_select": false,
                "context": {"tool": "send_email", "args": {...}}
            }
        ]
    }

Resume payload::

    Command(resume={
        "answers": {"Send email?": "Approve"},
        "edited_args": {...}  # optional, only for Edit decisions
    })
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.types import interrupt

from langchain_agentkit.extension import Extension
from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool
from langchain_agentkit.extensions.hitl.types import Option, Question

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools import BaseTool

DecisionType = Literal["approve", "edit", "reject"]

_DECISION_OPTIONS: dict[DecisionType, Option] = {
    "approve": Option(label="Approve", description="Execute the tool call as-is"),
    "edit": Option(label="Edit", description="Modify the arguments before executing"),
    "reject": Option(label="Reject", description="Deny this tool call"),
}


class InterruptConfig:
    """Configuration for a tool approval question.

    Defines which options to present and the question text shown to the
    human reviewer. Vocabulary aligns with the Question model.

    Args:
        options: Which approval options to present (approve, edit, reject).
        question: Static string or callable that generates the question
            text. Callable receives ``(tool_call,)``. Defaults to a
            summary of the tool name and arguments.
    """

    __slots__ = ("options", "question")

    def __init__(
        self,
        *,
        options: list[DecisionType],
        question: str | Callable[..., str] | None = None,
    ) -> None:
        self.options = options
        self.question = question


class HITLExtension(Extension):
    """Human-in-the-loop extension with unified Question protocol.

    Provides tool approval via a ``wrap_tool`` hook and an optional
    ``ask_user`` tool for LLM-initiated questions. Both use the same
    Question-based interrupt payload.

    The ``wrap_tool`` hook composes with other extensions' tool hooks
    via the onion pattern — multiple extensions can each wrap tool
    execution without conflict.

    Args:
        interrupt_on: Whitelist of tool names to gate with human approval.
            Only tools listed here will be interrupted — unlisted tools
            execute normally.

            - ``True``: all options (approve, edit, reject)
            - ``dict``: config with ``options`` and optional ``question``
            - ``InterruptConfig``: full config object

        tools: Whether to provide the ``ask_user`` tool to the LLM.

    Example::

        hitl = HITLExtension(
            interrupt_on={
                "send_email": True,
                "delete_file": {"options": ["approve", "reject"]},
            },
            tools=True,
        )

        class MyAgent(Agent):
            model = ChatOpenAI(model="gpt-4o")
            tools = [send_email]
            extensions = [hitl]

            async def handler(state, *, llm, tools, prompt, runtime):
                ...

        graph = MyAgent().compile().compile(checkpointer=InMemorySaver())
    """

    def __init__(
        self,
        *,
        interrupt_on: dict[str, bool | dict[str, Any] | InterruptConfig] | None = None,
        tools: bool = False,
    ) -> None:
        resolved: dict[str, InterruptConfig] = {}
        for tool_name, config in (interrupt_on or {}).items():
            if config is True:
                resolved[tool_name] = InterruptConfig(
                    options=["approve", "edit", "reject"],
                )
            elif isinstance(config, InterruptConfig):
                resolved[tool_name] = config
            elif isinstance(config, dict) and config.get("options"):
                resolved[tool_name] = InterruptConfig(
                    options=config["options"],
                    question=config.get("question"),
                )
            # False or unrecognized values are silently ignored — only
            # whitelisted tools get interrupted.
        self.interrupt_on = resolved
        self._provide_tools = tools
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> list[BaseTool]:
        """Returns ask_user tool when enabled, empty list otherwise."""
        if self._tools_cache is None:
            self._tools_cache = [create_ask_user_tool()] if self._provide_tools else []
        return self._tools_cache

    async def wrap_tool(
        self,
        *,
        state: Any,
        handler: Callable[..., Any],
        runtime: Any,
    ) -> Any:
        """Intercept tool calls and present structured approval questions.

        This is a ``wrap_tool`` hook — it composes with other extensions'
        tool hooks via the onion pattern. Unconfigured tools pass through
        to the inner handler without interruption.

        Uses the unified Question protocol. The interrupt payload contains
        a Question with Approve/Edit/Reject options and tool context.

        Resume payload::

            {"answers": {"<question>": "Approve"}}
            {"answers": {"<question>": "Edit"}, "edited_args": {...}}
            {"answers": {"<question>": "Reject"}, "message": "reason"}

        Args:
            state: The ``ToolCallRequest`` from the tool node.
            handler: Async callback to continue the tool execution chain.
            runtime: The current ``ToolRuntime``.
        """
        request = state
        tool_name = (
            request.tool_call.get("name", "")
            if isinstance(request.tool_call, dict)
            else getattr(request.tool_call, "name", "")
        )
        config = self.interrupt_on.get(tool_name)

        if config is None:
            return await handler(request)

        question_options = [_DECISION_OPTIONS[d] for d in config.options if d in _DECISION_OPTIONS]

        # Single option — auto-execute without interrupting
        if len(question_options) < 2:
            return await self._auto_execute(config, request, handler)

        question_text = self._build_question_text(request, config)
        question = Question(
            question=question_text,
            header=tool_name[:12],
            options=question_options,
            context={"tool": tool_name, "args": request.tool_call["args"]},
        )

        response = interrupt(
            {
                "type": "question",
                "questions": [question.model_dump()],
            }
        )

        return await self._handle_response(response, question, config, request, handler)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _auto_execute(
        self,
        config: InterruptConfig,
        request: Any,
        handler: Callable[..., Any],
    ) -> Any:
        """Handle single-option configs without interrupting."""
        decision = config.options[0]
        if decision == "approve":
            return await handler(request)
        tool_name = request.tool_call["name"]
        return ToolMessage(
            content=f"Auto-rejected {tool_name} (only allowed option: {decision})",
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    async def _handle_response(  # noqa: PLR0911
        self,
        response: Any,
        question: Question,
        config: InterruptConfig,
        request: Any,
        handler: Callable[..., Any],
    ) -> Any:
        """Route the user's answer to the appropriate action."""
        answers: dict[str, str] = {}
        if isinstance(response, dict):
            answers = response.get("answers", {})
        answer = answers.get(question.question, "")
        tool_name = request.tool_call["name"]

        if answer == "Approve" and "approve" in config.options:
            return await handler(request)

        if answer == "Edit" and "edit" in config.options:
            edited_args = (
                response.get("edited_args", request.tool_call["args"])
                if isinstance(response, dict)
                else request.tool_call["args"]
            )
            modified_call = ToolCall(
                name=tool_name,
                args=edited_args,
                id=request.tool_call["id"],
                type="tool_call",
            )
            return await handler(request.override(tool_call=modified_call))

        if answer == "Reject" and "reject" in config.options:
            message = (
                response.get("message", f"User rejected {tool_name}")
                if isinstance(response, dict)
                else f"User rejected {tool_name}"
            )
            return ToolMessage(
                content=message,
                name=tool_name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )

        return ToolMessage(
            content=f"Invalid answer '{answer}' for {tool_name}. Allowed: {config.options}",
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def _build_question_text(
        self,
        request: Any,
        config: InterruptConfig,
    ) -> str:
        """Build the question text for the interrupt."""
        if config.question is None:
            tool_name = request.tool_call["name"]
            tool_args = request.tool_call["args"]
            return f"Tool: {tool_name}\nArgs: {tool_args}"
        if callable(config.question):
            return config.question(request.tool_call)
        return config.question
