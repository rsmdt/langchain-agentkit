"""HITLExtension — human-in-the-loop via a unified Question protocol.

Two capabilities:

1. **Tool approval** — a dedicated node between the agent and the ToolNode
   batches every gated call in a step into one ``interrupt()`` and applies the
   human's decision (approve / edit / reject / respond) before any tool runs.
2. **AskUser tool** — lets the LLM ask the user structured questions.

The approval interrupt runs in the ``before_tools`` hook (its own node, not a
per-tool ``wrap_tool``) so a single batched interrupt covers all gated calls in
a step: this is replay-safe — the model call sits in an earlier super-step and
is not re-run — and avoids the per-tool interrupt-id collision that misroutes
resume values. Reject and respond become the call's ``ToolMessage`` right there,
carrying the decision in ``artifact``; because ``ToolNode`` executes every call
on the ``AIMessage``, ``wrap_tool`` returns that existing message instead of
running the tool. Approve runs as-is and edit rewrites the call's args.

Resume with ``answers`` keyed by each gated call's position in the step::

    Command(resume={"answers": {
        "0": "Approve",                          # or "Reject" — an option label
        "1": {"type": "edit", "args": {...}},    # edit/respond carry data
    }})

A missing or unrecognized answer denies the call.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Literal, cast, override

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.hitl.tools import create_ask_user_tool
from langchain_agentkit.extensions.hitl.types import Option, Question

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.tools import BaseTool

DecisionType = Literal["approve", "edit", "reject", "respond"]

_DECISION_OPTIONS: dict[DecisionType, Option] = {
    "approve": Option(label="Approve", description="Execute the tool call as-is"),
    "edit": Option(label="Edit", description="Modify the tool arguments before executing"),
    "reject": Option(label="Reject", description="Deny this tool call"),
    "respond": Option(
        label="Respond",
        description="Answer on behalf of the tool without executing it",
    ),
}

# Resume answers reference an option by label; map the label back to the
# decision it represents (case-insensitive).
_LABEL_TO_DECISION: dict[str, DecisionType] = {
    opt.label.lower(): decision for decision, opt in _DECISION_OPTIONS.items()
}


class InterruptConfig:
    """Configuration for a tool approval question.

    Defines which decisions to offer and the question text shown to the human
    reviewer. Vocabulary aligns with the Question model.

    Args:
        options: Which decisions to present (approve, edit, reject, respond).
        question: Static string or callable that generates the question text.
            Callable receives ``(tool_call,)``. Defaults to a summary of the
            tool name and arguments.
        when: Optional predicate ``(tool_call, state) -> bool`` (sync or
            async). The call is held for approval only when it returns true;
            otherwise it runs without asking.
        redact: Optional ``(args) -> args`` applied to the arguments shown in
            the question and its context, never to the arguments the tool runs
            with.
        context: Optional ``(tool_call) -> dict`` merged into the question's
            ``context`` so a client can render the request (e.g. a connect
            prompt) without parsing the question text.
    """

    __slots__ = ("context", "options", "question", "redact", "when")

    def __init__(
        self,
        *,
        options: list[DecisionType],
        question: str | Callable[..., str] | None = None,
        when: Callable[[ToolCall, dict[str, Any]], bool | Awaitable[bool]] | None = None,
        redact: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        context: Callable[[ToolCall], dict[str, Any]] | None = None,
    ) -> None:
        self.options = options
        self.question = question
        self.when = when
        self.redact = redact
        self.context = context


class HITLExtension(Extension):
    """Human-in-the-loop extension with unified Question protocol.

    Tool approval runs in a dedicated node placed between the agent node and
    the ToolNode: it batches every gated call in the step into one
    ``interrupt()`` and applies the human's decisions (approve / edit / reject
    / respond) before any tool executes. An optional ``AskUser`` tool lets the
    LLM ask structured questions. Both share the Question-based payload.

    Args:
        interrupt_on: Whitelist of tool names to gate with human approval.
            Only tools listed here are interrupted — unlisted tools execute
            normally.

            - ``True``: all decisions (approve, edit, reject, respond)
            - ``dict``: config with ``options`` and optional ``question``
            - ``InterruptConfig``: full config object

        tools: Optional explicit tool list. When ``None`` (default), the
            extension provides ``[AskUser]``. Pass ``[]`` to disable the
            AskUser tool entirely, or pass a custom tool list to replace the
            default.

    Example::

        hitl = HITLExtension(
            interrupt_on={
                "send_email": True,
                "delete_file": {"options": ["approve", "reject"]},
            },
        )

        class MyAgent(Agent):
            model = ChatOpenAI(model="gpt-4o")
            tools = [send_email]
            extensions = [hitl]

            async def handler(state, *, llm, tools, prompt, runtime):
                ...

        graph = await MyAgent().compile(checkpointer=InMemorySaver())
    """

    def __init__(
        self,
        *,
        interrupt_on: dict[str, bool | dict[str, Any] | InterruptConfig] | None = None,
        tools: Sequence[BaseTool] | None = None,
    ) -> None:
        resolved: dict[str, InterruptConfig] = {}
        for tool_name, config in (interrupt_on or {}).items():
            if config is True:
                resolved[tool_name] = InterruptConfig(
                    options=["approve", "edit", "reject", "respond"],
                )
            elif isinstance(config, InterruptConfig):
                resolved[tool_name] = config
            elif isinstance(config, dict) and config.get("options"):
                resolved[tool_name] = InterruptConfig(
                    options=config["options"],
                    question=config.get("question"),
                    when=config.get("when"),
                    redact=config.get("redact"),
                    context=config.get("context"),
                )
            # False or unrecognized values are silently ignored — only
            # whitelisted tools get interrupted.
        self.interrupt_on = resolved
        self._custom_tools: tuple[BaseTool, ...] | None = (
            tuple(tools) if tools is not None else None
        )
        self._tools_cache: list[BaseTool] | None = None

    @property
    @override
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            if self._custom_tools is not None:
                self._tools_cache = list(self._custom_tools)
            else:
                self._tools_cache = [create_ask_user_tool()]
        return self._tools_cache

    # ------------------------------------------------------------------
    # Approval — one batched interrupt over every gated call, before tools run
    # ------------------------------------------------------------------

    async def before_tools(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls),
            None,
        )
        if last_ai is None:
            return None
        gated = await self._gated_calls(last_ai, state)
        if not gated:
            return None

        answers: list[ToolMessage] = []
        edits: dict[str, dict[str, Any]] = {}
        questions: list[Question] = []
        interrupt_calls: list[tuple[ToolCall, InterruptConfig]] = []

        for tc, config in gated:
            options = [_DECISION_OPTIONS[d] for d in config.options if d in _DECISION_OPTIONS]
            if len(options) < 2:
                self._apply_decision(
                    {"type": config.options[0] if config.options else "reject"},
                    tc,
                    answers,
                    edits,
                )
                continue
            shown_args = config.redact(tc["args"]) if config.redact else tc["args"]
            questions.append(
                Question(
                    question=self._question_text({**tc, "args": shown_args}, config),
                    header=tc["name"][:12],
                    options=options,
                    context={
                        **(config.context(tc) if config.context else {}),
                        "tool": tc["name"],
                        "args": shown_args,
                        "allowed": list(config.options),
                    },
                )
            )
            interrupt_calls.append((tc, config))

        if questions:
            response = interrupt(
                {
                    "type": "question",
                    "questions": [q.model_dump() for q in questions],
                }
            )
            given = response.get("answers") or {} if isinstance(response, dict) else {}
            for i, (tc, config) in enumerate(interrupt_calls):
                decision = self._coerce_decision(given.get(str(i)), config)
                self._apply_decision(decision, tc, answers, edits)

        result: list[Any] = []
        if edits:
            last_ai.tool_calls = [
                cast("ToolCall", {**tc, "args": edits[tc["id"]]}) if tc["id"] in edits else tc
                for tc in last_ai.tool_calls
            ]
            result.append(last_ai)
        result.extend(answers)
        return {"messages": result} if result else None

    # ------------------------------------------------------------------
    # Tool execution — substitute results for non-executing decisions
    # ------------------------------------------------------------------

    async def wrap_tool(
        self,
        *,
        state: Any,
        handler: Callable[..., Any],
        runtime: Any,
    ) -> Any:
        request = state
        tool_call = request.tool_call
        call_id = (
            tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
        )
        graph_state = getattr(request, "state", None) or {}
        existing = _answers_to_last_request(graph_state.get("messages", [])).get(str(call_id))
        if existing is not None:
            return existing
        return await handler(request)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coerce_decision(self, answer: Any, config: InterruptConfig) -> dict[str, Any]:
        """Normalize a resume answer into a typed decision dict.

        Accepts a decision dict (``{"type": ...}``) verbatim, or an option
        label string for the common approve/reject case. A missing or
        unrecognized answer denies the call — the safe default for gating.
        """
        if isinstance(answer, dict) and answer.get("type") in config.options:
            return answer
        if isinstance(answer, str):
            decision = _LABEL_TO_DECISION.get(answer.strip().lower())
            if decision in ("approve", "reject") and decision in config.options:
                return {"type": decision}
        return {"type": "reject", "message": "(no response)"}

    async def _gated_calls(
        self, last_ai: AIMessage, state: dict[str, Any]
    ) -> list[tuple[ToolCall, InterruptConfig]]:
        answered = _answers_to_last_request(state.get("messages", []))
        gated: list[tuple[ToolCall, InterruptConfig]] = []
        for tc in last_ai.tool_calls:
            config = self.interrupt_on.get(tc["name"])
            if config is None or tc["id"] in answered:
                continue
            if await self._holds(config, tc, state):
                gated.append((tc, config))
        return gated

    async def _holds(
        self, config: InterruptConfig, tool_call: ToolCall, state: dict[str, Any]
    ) -> bool:
        if config.when is None:
            return True
        held = config.when(tool_call, state)
        if inspect.isawaitable(held):
            held = await held
        return bool(held)

    def _apply_decision(
        self,
        decision: dict[str, Any],
        tool_call: ToolCall,
        answers: list[ToolMessage],
        edits: dict[str, dict[str, Any]],
    ) -> None:
        """Route a decision: edit rewrites args, reject/respond answer the call."""
        dtype = decision.get("type", "reject")
        call_id = cast("str", tool_call["id"])
        if dtype == "approve":
            return
        if dtype == "edit":
            edits[call_id] = (
                decision.get("args")
                or decision.get("edited_action", {}).get("args")
                or tool_call["args"]
            )
            return
        if dtype == "respond":
            answers.append(
                ToolMessage(
                    content=decision.get("message", ""),
                    name=tool_call["name"],
                    tool_call_id=call_id,
                    status="success",
                    artifact={"decision": {"type": "respond"}},
                )
            )
            return
        answers.append(
            ToolMessage(
                content=decision.get("message")
                or f"User rejected the {tool_call['name']} tool call.",
                name=tool_call["name"],
                tool_call_id=call_id,
                status="error",
                artifact={"decision": {"type": "reject"}},
            )
        )

    def _question_text(self, tool_call: ToolCall, config: InterruptConfig) -> str:
        if config.question is None:
            return f"Tool: {tool_call['name']}\nArgs: {tool_call['args']}"
        if callable(config.question):
            return config.question(tool_call)
        return config.question


def _answers_to_last_request(messages: list[Any]) -> dict[str, ToolMessage]:
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], AIMessage) and messages[i].tool_calls
        ),
        -1,
    )
    return {m.tool_call_id: m for m in messages[start + 1 :] if isinstance(m, ToolMessage)}
