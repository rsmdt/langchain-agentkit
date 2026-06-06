"""Verdict types and the grader's structured-output schema.

These are pure data definitions with no framework coupling — the grader
emits a :class:`GraderResponse`, the middleware records a
:class:`RubricEvaluation` per iteration, and the verdict literals drive the
loop's control flow.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, model_validator
from typing_extensions import TypedDict

GraderVerdict = Literal["satisfied", "needs_revision", "failed"]
"""Verdict the grader sub-agent emits via structured output.

- ``satisfied``: every criterion passes.
- ``needs_revision``: at least one criterion fails; loop continues.
- ``failed``: the rubric itself is malformed or impossible to evaluate
  against the transcript.
"""

RubricResult = GraderVerdict | Literal["review_limit_reached", "grader_error"]
"""Status recorded on each evaluation.

Superset of :data:`GraderVerdict` with two extension-synthesized terminal
statuses the grader cannot emit itself:

- ``review_limit_reached``: the review budget (``review.stop``) was exhausted
  on a ``needs_revision`` verdict; the agent terminates with its last response
  intact.
- ``grader_error``: the grader sub-agent raised an exception (provider
  timeout, missing credentials, malformed structured response, etc.).
  Distinct from ``failed``, which the grader returns about the *rubric*,
  not about its own machinery.

Only ``needs_revision`` continues the loop; every other status ends the
grading run.
"""

TERMINAL_RESULTS: frozenset[RubricResult] = frozenset(
    {"satisfied", "review_limit_reached", "failed", "grader_error"}
)
"""Statuses that signal a completed grading run; a same-rubric invocation
after one of these starts a fresh run with a new ``grading_run_id`` and a
reset review budget."""

RUBRIC_GRADER_MESSAGE_SOURCE = "rubric_grader"
"""Tag stored on synthetic revision messages this extension injects.

The revision message is injected as a ``HumanMessage`` (the role the model
follows most reliably), but it carries:

- ``name="rubric_grader"`` — visible at the wire on providers that
  round-trip the ``name`` field; ignored elsewhere.
- ``additional_kwargs={"lc_source": RUBRIC_GRADER_MESSAGE_SOURCE}`` —
  visible to in-process consumers (evals, UIs, observability) so they can
  attribute the turn to the grader instead of treating it as a real user
  message.
"""


class CriterionPass(TypedDict):
    """Per-criterion grader verdict when the criterion passes."""

    name: str
    """Short label identifying the criterion (e.g., the rubric bullet)."""

    passed: Literal[True]
    """Discriminator: this verdict variant has no ``gap``."""


class CriterionFail(TypedDict):
    """Per-criterion grader verdict when the criterion fails."""

    name: str
    """Short label identifying the criterion (e.g., the rubric bullet)."""

    passed: Literal[False]
    """Discriminator: this verdict variant requires ``gap``."""

    gap: str
    """Short, actionable description of what's missing or incorrect."""


CriterionEval = Annotated[CriterionPass | CriterionFail, Discriminator("passed")]
"""Per-criterion verdict.

Discriminated union on ``passed``: pass-verdicts have no ``gap``;
fail-verdicts require one. :meth:`GraderResponse.model_validate` enforces the
shape at the trust boundary so a grader cannot emit ``{passed: True, gap: ...}``
or ``{passed: False}`` with no gap.
"""


class RubricEvaluation(TypedDict):
    """One grader evaluation, appended to ``_rubric_evaluations`` each iteration.

    Consumers can read any field without guarding against absence since all
    fields are always populated by the extension.
    """

    grading_run_id: str
    """Identifier shared by all evaluations within a single grading run.

    A new run starts when the caller supplies a different rubric, or when
    the same rubric is re-invoked after a terminal verdict.
    """

    iteration: int
    """Zero-based index within the current rubric attempt."""

    result: RubricResult
    """The grader's terminal verdict for this iteration."""

    explanation: str
    """Free-form summary of the verdict, from the grader."""

    criteria: list[CriterionEval]
    """Per-criterion verdicts."""


class GraderResponse(BaseModel):
    """Structured output the grader sub-agent must emit.

    Passed as ``response_format=GraderResponse`` to the grader agent so the
    underlying provider's structured output strategy is auto-selected.
    """

    result: GraderVerdict = Field(
        description=(
            "Terminal verdict for this evaluation. Use 'satisfied' only when every "
            "criterion passes; 'needs_revision' when at least one criterion fails; "
            "'failed' when the rubric cannot be evaluated."
        ),
    )
    explanation: str = Field(
        description=(
            "One or two sentence verdict summary that will be sent back to the agent "
            "as feedback if the task needs to be reattempted."
        ),
    )
    criteria: list[CriterionEval] = Field(
        default_factory=list,
        description=(
            "Per-criterion verdicts. Each criterion should appear once with `passed` "
            "True/False and a `gap` string when failing."
        ),
    )

    @model_validator(mode="after")
    def _check_result_consistency(self) -> GraderResponse:
        """Reject grader output where ``result`` contradicts the criteria.

        The grader is an LLM and can hallucinate self-inconsistent
        responses (e.g. claiming ``satisfied`` while flagging a failing
        criterion). The discriminated union on :data:`CriterionEval`
        enforces the per-criterion ``gap`` invariant; this validator catches
        the cross-field one.
        """
        has_fail = any(not c["passed"] for c in self.criteria)
        if self.result == "satisfied" and has_fail:
            msg = "GraderResponse: result='satisfied' but at least one criterion has passed=False."
            raise ValueError(msg)
        if self.result == "needs_revision" and self.criteria and not has_fail:
            msg = "GraderResponse: result='needs_revision' but every criterion has passed=True."
            raise ValueError(msg)
        return self
