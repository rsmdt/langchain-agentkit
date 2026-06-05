"""RubricExtension — self-evaluated iteration against a caller-supplied rubric.

A caller declares *what done looks like* by passing a ``rubric`` on the
invocation state. Each time the agent would otherwise finish — the model
returns a response with no further tool calls — the extension grades the
transcript with a separate grader sub-agent. On ``needs_revision`` the
grader's per-criterion feedback is injected as a ``HumanMessage`` and the
loop resumes at the model; grading repeats until ``satisfied`` or ``failed``,
or ``max_iterations`` is reached.

With no ``rubric`` on state both hooks are no-ops, so the extension is safe
to include unconditionally.

The revision loop relies on routing back to the model via ``jump_to``, which
is only honored when the kit exposes at least one tool (a tool-less kit wires
the model node straight to the end). Rubric use cases — run tests, lint, edit
files — always carry tools; a tool-less kit grades once without looping.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, override

from langchain_core.messages import HumanMessage

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.rubric.grader import (
    GRADER_SYSTEM_PROMPT,
    build_grader,
    build_grader_payload,
    extract_graded,
)
from langchain_agentkit.extensions.rubric.state import RubricState
from langchain_agentkit.extensions.rubric.types import (
    RUBRIC_GRADER_MESSAGE_SOURCE,
    TERMINAL_RESULTS,
    GraderResponse,
    GraderVerdict,
    RubricEvaluation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_MAX_ITERATIONS_HARD_CAP = 20
"""Hard upper bound for ``max_iterations``."""


class RubricExtension(Extension):
    """Drive self-evaluated iteration against a rubric.

    Args:
        model: Model used by the grader sub-agent — a ``"provider:model-id"``
            string or a ``BaseChatModel`` instance. A string is resolved via
            the kit's ``model_resolver``; pass a ``BaseChatModel`` directly if
            the kit has no resolver.
        system_prompt: Custom grading instructions; falls back to the
            built-in grader prompt when not set.
        tools: Tools the grader may call before producing its verdict (e.g. a
            test runner). With none, the grader reasons from the transcript
            alone.
        max_iterations: Hard cap on grader iterations per rubric attempt;
            hard-capped at 20. When the cap is reached without a ``satisfied``
            verdict, the run terminates with status
            ``'max_iterations_reached'`` and a ``logger.warning``.
        on_evaluation: Optional callback invoked with each
            :class:`RubricEvaluation` after grading. Exceptions it raises are
            logged and suppressed; do not use it to enforce control flow.

    Raises:
        ValueError: If ``max_iterations`` is outside ``[1, 20]``, or ``model``
            is falsy.
        TypeError: If ``max_iterations`` is not an ``int``.
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        system_prompt: str | None = None,
        tools: Sequence[BaseTool] | None = None,
        max_iterations: int = 3,
        on_evaluation: Callable[[RubricEvaluation], None] | None = None,
    ) -> None:
        if not model:
            msg = "RubricExtension: `model` is required."
            raise ValueError(msg)
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            msg = (
                "RubricExtension: `max_iterations` must be an int, "
                f"got {type(max_iterations).__name__}."
            )
            raise TypeError(msg)
        if not 1 <= max_iterations <= _MAX_ITERATIONS_HARD_CAP:
            msg = (
                f"RubricExtension: `max_iterations` must be in "
                f"[1, {_MAX_ITERATIONS_HARD_CAP}], got {max_iterations}."
            )
            raise ValueError(msg)

        self.max_iterations = max_iterations
        self._model = model
        self._system_prompt = system_prompt or GRADER_SYSTEM_PROMPT
        self._tools: list[BaseTool] = list(tools) if tools else []
        self._on_evaluation = on_evaluation
        self._model_resolver: Callable[[str], BaseChatModel] | None = None
        # Built lazily so constructing the extension doesn't construct a model
        # client (which can trigger env-var lookups / API key validation).
        self._grader: Any = None

    @override
    def setup(self, *, model_resolver: Any = None, **_: Any) -> None:
        # Capture the kit's resolver so a string grader model can be resolved
        # lazily, consistent with how the main agent model is resolved.
        self._model_resolver = model_resolver

    @property
    @override
    def state_schema(self) -> type:
        return RubricState

    # --- Lifecycle hooks ---

    async def before_run(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Detect a new grading run and reset iteration bookkeeping.

        A "new grading run" is either a different ``rubric`` string than
        ``_active_rubric``, or the same ``rubric`` after the previous run
        reached a terminal status. With no ``rubric`` the extension is a
        no-op for this run.
        """
        return self._reset_for_new_rubric(state)

    async def after_model(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Grade the transcript and decide whether to loop back to the model.

        Runs only at a natural stop — when the model's latest message carries
        no tool calls, i.e. the agent would otherwise terminate this turn.
        May return ``jump_to='model'`` with an injected revision message to
        loop, or omit ``jump_to`` to let the run end.
        """
        rubric = state.get("rubric")
        if not rubric:
            return None
        messages = state.get("messages") or []
        if not _is_natural_stop(messages):
            return None

        iteration = state.get("_rubric_iterations", 0) or 0
        grading_run_id = state.get("_current_grading_run_id") or str(uuid.uuid4())
        self._emit(runtime, "rubric_evaluation_start", grading_run_id, iteration)

        try:
            graded = await self._grade(rubric, messages, iteration)
        except Exception as exc:  # noqa: BLE001
            return self._handle_grader_exception(runtime, state, grading_run_id, iteration, exc)

        evaluation = _build_evaluation(graded, grading_run_id, iteration)
        self._emit(runtime, "rubric_evaluation_end", grading_run_id, iteration, evaluation)
        self._run_callback(evaluation)
        return self._compose_update(state, evaluation, graded.result)

    # --- Reset / grading internals ---

    def _reset_for_new_rubric(self, state: dict[str, Any]) -> dict[str, Any] | None:
        rubric = state.get("rubric")
        if not rubric:
            return None
        same_rubric = state.get("_active_rubric") == rubric
        previous_terminal = state.get("_rubric_status") in TERMINAL_RESULTS
        if same_rubric and not previous_terminal:
            return None
        return {
            "_rubric_iterations": 0,
            "_rubric_status": None,
            "_current_grading_run_id": str(uuid.uuid4()),
            "_active_rubric": rubric,
        }

    def _ensure_grader(self) -> Any:
        if self._grader is None:
            self._grader = build_grader(
                model=self._resolve_grader_model(),
                system_prompt=self._system_prompt,
                tools=self._tools,
            )
        return self._grader

    def _resolve_grader_model(self) -> Any:
        model = self._model
        if not isinstance(model, str):
            return model
        if self._model_resolver is not None:
            return self._model_resolver(model)
        msg = (
            "RubricExtension: grader `model` is a string but the kit has no "
            "model_resolver. Pass model_resolver=<callable> to AgentKit, or "
            "give RubricExtension a BaseChatModel instance."
        )
        raise ValueError(msg)

    async def _grade(
        self, rubric: str, messages: list[AnyMessage], iteration: int
    ) -> GraderResponse:
        grader = self._ensure_grader()
        payload = build_grader_payload(rubric, messages, iteration)
        result = await grader.ainvoke({"messages": [HumanMessage(content=payload)]})
        return extract_graded(result)

    def _compose_update(
        self,
        state: dict[str, Any],
        evaluation: RubricEvaluation,
        graded_result: GraderVerdict,
    ) -> dict[str, Any]:
        iteration = evaluation["iteration"]
        next_iteration = iteration + 1
        evals = [*state.get("_rubric_evaluations", []), evaluation]

        update: dict[str, Any] = {
            "_rubric_evaluations": evals,
            "_rubric_iterations": next_iteration,
            "_rubric_status": evaluation["result"],
        }

        if graded_result == "satisfied":
            return update

        if graded_result == "failed":
            update["_rubric_status"] = "failed"
            return update

        # needs_revision
        if next_iteration >= self.max_iterations:
            logger.warning(
                "RubricExtension exhausted max_iterations=%d without 'satisfied' "
                "verdict (grading_run_id=%s)",
                self.max_iterations,
                evaluation["grading_run_id"],
            )
            update["_rubric_status"] = "max_iterations_reached"
            return update

        return {
            **update,
            "messages": [
                HumanMessage(
                    content=_revision_prompt(evaluation),
                    name=RUBRIC_GRADER_MESSAGE_SOURCE,
                    additional_kwargs={"lc_source": RUBRIC_GRADER_MESSAGE_SOURCE},
                )
            ],
            "jump_to": "model",
        }

    def _handle_grader_exception(
        self,
        runtime: Any,
        state: dict[str, Any],
        grading_run_id: str,
        iteration: int,
        exc: Exception,
    ) -> dict[str, Any]:
        logger.exception("RubricExtension grader failed")
        evaluation: RubricEvaluation = {
            "grading_run_id": grading_run_id,
            "iteration": iteration,
            "result": "grader_error",
            "explanation": f"Grader raised {type(exc).__name__}: {exc}",
            "criteria": [],
        }
        self._emit(runtime, "rubric_evaluation_end", grading_run_id, iteration, evaluation)
        self._run_callback(evaluation)
        evals = [*state.get("_rubric_evaluations", []), evaluation]
        return {
            "_rubric_evaluations": evals,
            "_rubric_iterations": iteration + 1,
            "_rubric_status": "grader_error",
        }

    def _run_callback(self, evaluation: RubricEvaluation) -> None:
        if self._on_evaluation is None:
            return
        try:
            self._on_evaluation(evaluation)
        except Exception:
            logger.exception("RubricExtension on_evaluation callback raised")

    def _emit(
        self,
        runtime: Any,
        event_type: str,
        grading_run_id: str,
        iteration: int,
        evaluation: RubricEvaluation | None = None,
    ) -> None:
        writer = getattr(runtime, "stream_writer", None)
        if writer is None:
            return
        payload: dict[str, Any] = {
            "type": event_type,
            "grading_run_id": grading_run_id,
            "iteration": iteration,
        }
        if evaluation is not None:
            payload["result"] = evaluation.get("result")
            payload["explanation"] = evaluation.get("explanation")
            payload["criteria"] = evaluation.get("criteria", [])
        try:
            writer(payload)
        except Exception:  # noqa: BLE001
            logger.debug("RubricExtension stream_writer raised; ignoring")


def _is_natural_stop(messages: list[AnyMessage]) -> bool:
    """True when the model's latest message carries no tool calls.

    Mirrors the graph's own termination test: a trailing assistant message
    with tool calls routes to the tools node, otherwise the loop ends. The
    grader fires only on the would-terminate turn.
    """
    if not messages:
        return False
    last = messages[-1]
    return not (getattr(last, "tool_calls", None))


def _build_evaluation(
    graded: GraderResponse,
    grading_run_id: str,
    iteration: int,
) -> RubricEvaluation:
    return {
        "grading_run_id": grading_run_id,
        "iteration": iteration,
        "result": graded.result,
        "explanation": graded.explanation,
        "criteria": [dict(c) for c in graded.criteria],  # type: ignore[misc]
    }


def _revision_prompt(evaluation: RubricEvaluation) -> str:
    lines = [
        "A grader reviewed your work against the rubric and asked for "
        "revisions before we can finish."
    ]
    explanation = evaluation.get("explanation")
    if explanation:
        lines.append("")
        lines.append(f"Grader feedback: {explanation.strip()}")

    failing = [c for c in evaluation.get("criteria", []) if not c.get("passed")]
    if failing:
        lines.append("")
        lines.append("Criteria that still need work:")
        for criterion in failing:
            name = criterion.get("name", "(unnamed criterion)")
            gap = str(criterion.get("gap") or "").strip()
            if gap:
                lines.append(f"- {name}: {gap}")
            else:
                lines.append(f"- {name} (no specific feedback provided)")

    lines.append("")
    lines.append(
        "Please address every failing criterion and respond when you believe "
        "the rubric is satisfied."
    )
    return "\n".join(lines)
