"""RubricExtension — rubric-gated iteration in two modes.

A caller declares *what done looks like* by passing a ``rubric`` on the
invocation state. A separate grader sub-agent reviews the work against it and
returns a per-criterion verdict. What happens on ``needs_revision`` depends on
``mode``:

- ``auto`` — the closed loop: the grader's feedback is injected and the model
  re-drafts autonomously (no user), until ``satisfied`` / ``failed`` or the
  review budget runs out. The agent never sees the rubric — the grader is an
  independent verifier.
- ``user`` — the open loop: the turn ends and control returns to the user. The
  agent *does* see the rubric and the still-failing gaps, so it steers its
  prose (or, if a HITL tool is present, structured questions) toward the
  criteria across normal conversation turns.

``review`` (a :class:`ReviewPolicy`) controls *when* the grader runs: a turn
window ``[min, max]`` with a give-up budget ``stop``. In ``user`` mode with a
window, the agent gets a ``request_review`` tool to be graded early.

With no ``rubric`` on state every hook is a no-op, so the extension is safe to
include unconditionally.

The ``auto`` revision loop relies on routing back to the model via ``jump_to``,
honored only when the kit exposes at least one tool. ``user`` mode has no such
requirement — it ends the turn rather than looping.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Literal, override

from langchain_core.messages import HumanMessage

from langchain_agentkit.composition.extension import Extension
from langchain_agentkit.extensions.rubric.grader import (
    GRADER_SYSTEM_PROMPT,
    build_grader,
    build_grader_payload,
    extract_graded,
)
from langchain_agentkit.extensions.rubric.policy import ReviewPolicy
from langchain_agentkit.extensions.rubric.state import RubricState
from langchain_agentkit.extensions.rubric.tools import create_request_review_tool
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

RubricMode = Literal["auto", "user"]


class RubricExtension(Extension):
    """Drive rubric-gated iteration, autonomously or with the user in the loop.

    Args:
        model: Model used by the grader sub-agent — a ``"provider:model-id"``
            string or a ``BaseChatModel`` instance. A string is resolved via
            the kit's ``model_resolver``; pass a ``BaseChatModel`` directly if
            the kit has no resolver.
        mode: ``"auto"`` (default) self-revises in a closed loop; ``"user"``
            ends the turn on ``needs_revision`` and surfaces the rubric + gaps
            to the agent so it collaborates with the user.
        review: When the grader runs — an ``int`` (``N`` → grade by turn N),
            a dict, or a :class:`ReviewPolicy`. In ``auto`` mode every draft is
            graded and only ``review.stop`` applies; the window matters in
            ``user`` mode.
        system_prompt: Custom grading instructions; falls back to the built-in
            grader prompt when not set.
        grader_tools: Tools the *grader* may call to gather evidence (e.g. a
            test runner). Distinct from the agent's tools. With none, the
            grader reasons from the transcript alone.
        on_evaluation: Optional callback invoked with each
            :class:`RubricEvaluation` after grading. Exceptions it raises are
            logged and suppressed; do not use it to enforce control flow.

    Raises:
        ValueError: If ``model`` is falsy, ``mode`` is unknown, or ``review``
            is out of range.
        TypeError: If ``review`` is not an int / dict / ReviewPolicy.
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel,
        mode: RubricMode = "auto",
        review: int | dict[str, int] | ReviewPolicy = 1,
        system_prompt: str | None = None,
        grader_tools: Sequence[BaseTool] | None = None,
        on_evaluation: Callable[[RubricEvaluation], None] | None = None,
    ) -> None:
        if not model:
            msg = "RubricExtension: `model` is required."
            raise ValueError(msg)
        if mode not in ("auto", "user"):
            msg = f"RubricExtension: `mode` must be 'auto' or 'user', got {mode!r}."
            raise ValueError(msg)

        self._mode: RubricMode = mode
        self._review = ReviewPolicy.coerce(review)
        self._model = model
        self._system_prompt = system_prompt or GRADER_SYSTEM_PROMPT
        self._grader_tools: list[BaseTool] = list(grader_tools) if grader_tools else []
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

    @property
    @override
    def tools(self) -> list[BaseTool]:
        # The agent gets request_review only when it has discretion over timing:
        # user mode with a real window. Otherwise grading cadence is fixed and
        # the tool would be meaningless.
        if self._mode == "user" and self._review.has_window:
            return [create_request_review_tool()]  # type: ignore[list-item]
        return []

    @override
    def prompt(
        self,
        state: dict[str, Any],
        runtime: Any | None = None,  # noqa: ARG002
        *,
        tools: frozenset[str] = frozenset(),  # noqa: ARG002
    ) -> str | dict[str, str] | None:
        # Only user mode surfaces the rubric to the agent — in auto mode the
        # grader is an independent verifier the agent must not see.
        if self._mode != "user":
            return None
        rubric = state.get("rubric")
        if not rubric:
            return None
        durable = (
            "Your work is reviewed against the rubric below. Work toward "
            "satisfying every criterion. If you need information to satisfy a "
            "criterion, ask the user. State clearly when you believe every "
            f"criterion is met.\n\n{str(rubric).strip()}"
        )
        reminder = _open_gaps_reminder(state.get("_rubric_evaluations") or [])
        return {"prompt": durable, "reminder": reminder} if reminder else durable

    # --- Lifecycle hooks ---

    async def before_run(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
        """Reset bookkeeping when a new grading run starts.

        A new run is a different ``rubric`` than ``_active_rubric``, or the
        same ``rubric`` after the previous run reached a terminal status. With
        no ``rubric`` the extension is a no-op for this run.
        """
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
            "_rubric_turns_since_review": 0,
            "_rubric_review_requested": False,
        }

    async def after_model(self, *, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Decide whether to grade this turn, and act on the verdict.

        Runs only at a natural stop (the model produced no tool calls). Whether
        grading happens is governed by the review window; what happens on
        ``needs_revision`` is governed by ``mode``.
        """
        rubric = state.get("rubric")
        if not rubric:
            return None
        if not _is_natural_stop(state.get("messages") or []):
            return None

        turns = (state.get("_rubric_turns_since_review", 0) or 0) + 1
        requested = bool(state.get("_rubric_review_requested"))
        if not self._should_grade(turns, requested):
            # Window not reached: in user mode the turn simply ends back to the
            # user; just carry the turn count forward.
            return {"_rubric_turns_since_review": turns}

        messages = state.get("messages") or []
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

    # --- Grading decision & action ---

    def _should_grade(self, turns_since_review: int, requested: bool) -> bool:
        """Whether to grade at this natural stop.

        ``auto`` grades every draft (the window is a ``user``-mode concept —
        each autonomous draft is a complete candidate). ``user`` grades when
        the forced bound is hit, or when the agent asked and the floor is met.
        """
        if self._mode == "auto":
            return True
        if turns_since_review >= self._review.max:
            return True
        return requested and turns_since_review >= self._review.min

    def _compose_update(
        self,
        state: dict[str, Any],
        evaluation: RubricEvaluation,
        graded_result: GraderVerdict,
    ) -> dict[str, Any]:
        next_iteration = evaluation["iteration"] + 1
        evals = [*state.get("_rubric_evaluations", []), evaluation]
        update: dict[str, Any] = {
            "_rubric_evaluations": evals,
            "_rubric_iterations": next_iteration,
            "_rubric_status": evaluation["result"],
            # A review just happened — reset the window counters.
            "_rubric_turns_since_review": 0,
            "_rubric_review_requested": False,
        }

        if graded_result == "satisfied" or graded_result == "failed":
            return update

        # needs_revision
        if next_iteration >= self._review.stop:
            logger.warning(
                "RubricExtension exhausted review budget (stop=%d) without "
                "'satisfied' verdict (grading_run_id=%s)",
                self._review.stop,
                evaluation["grading_run_id"],
            )
            update["_rubric_status"] = "review_limit_reached"
            return update

        if self._mode == "auto":
            # Closed loop: inject feedback and re-draft.
            update["messages"] = [
                HumanMessage(
                    content=_revision_prompt(evaluation),
                    name=RUBRIC_GRADER_MESSAGE_SOURCE,
                    additional_kwargs={"lc_source": RUBRIC_GRADER_MESSAGE_SOURCE},
                )
            ]
            update["jump_to"] = "model"
            return update

        # user mode: yield to the user. Gaps reach the agent via prompt() on
        # its next turn; no synthetic message, no jump. Status stays
        # ``needs_revision`` (non-terminal) so the run continues across turns.
        return update

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
            "_rubric_turns_since_review": 0,
            "_rubric_review_requested": False,
        }

    # --- Grader plumbing ---

    def _ensure_grader(self) -> Any:
        if self._grader is None:
            self._grader = build_grader(
                model=self._resolve_grader_model(),
                system_prompt=self._system_prompt,
                tools=self._grader_tools,
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


def _failing_criteria(evaluation: RubricEvaluation) -> list[dict[str, Any]]:
    return [dict(c) for c in evaluation.get("criteria", []) if not c.get("passed")]


def _open_gaps_reminder(evaluations: list[RubricEvaluation]) -> str:
    """Render the still-failing criteria from the latest review, for the agent.

    Returns ``""`` when the latest review passed or there are no evaluations —
    the agent only needs the gaps while they're open.
    """
    if not evaluations:
        return ""
    last = evaluations[-1]
    if last.get("result") != "needs_revision":
        return ""
    failing = _failing_criteria(last)
    if not failing:
        return ""
    lines = ["These rubric criteria are not yet satisfied:"]
    for criterion in failing:
        name = criterion.get("name", "(unnamed criterion)")
        gap = str(criterion.get("gap") or "").strip()
        lines.append(f"- {name}: {gap}" if gap else f"- {name}")
    return "\n".join(lines)


def _revision_prompt(evaluation: RubricEvaluation) -> str:
    lines = [
        "A grader reviewed your work against the rubric and asked for "
        "revisions before we can finish."
    ]
    explanation = evaluation.get("explanation")
    if explanation:
        lines.append("")
        lines.append(f"Grader feedback: {explanation.strip()}")

    failing = _failing_criteria(evaluation)
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
