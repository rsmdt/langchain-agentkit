"""State schema for :class:`RubricExtension`.

Only ``rubric`` is caller-facing — write a rubric, read the improved agent
response back from ``messages``. Everything else is bookkeeping the extension
owns: handlers must not write these keys directly. Each is a plain field
under LangGraph's default last-value semantics, which is correct because the
extension reads-then-writes them sequentially in ``before_run`` /
``after_model`` with no concurrent writer.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_agentkit.composition.state import PrivateStateAttr

# Imported at runtime (not under TYPE_CHECKING): LangGraph resolves this
# TypedDict's annotations via ``get_type_hints`` when building the graph, so
# the referenced types must be present in module globals.
from langchain_agentkit.extensions.rubric.types import (  # noqa: TC001
    RubricEvaluation,
    RubricResult,
)


class RubricState(TypedDict, total=False):
    """State mixin tracking the active rubric and grading bookkeeping.

    Only ``rubric`` is caller-facing. The ``_rubric_*`` keys are managed
    entirely by the extension and marked :class:`PrivateStateAttr`, so they
    stay out of the graph's public input/output schema. Read them via
    ``get_state(config).values`` on a checkpointed thread, the
    ``rubric_evaluation_*`` stream events, or the ``on_evaluation`` callback.
    """

    rubric: str
    """Caller-supplied rubric describing what ``done`` looks like."""

    _rubric_status: Annotated[RubricResult | None, PrivateStateAttr]
    """Most recent terminal status, or ``None`` after a fresh rubric attempt
    is started but before the first grader call."""

    _rubric_iterations: Annotated[int, PrivateStateAttr]
    """Grader evaluations performed for the current rubric attempt."""

    _rubric_evaluations: Annotated[list[RubricEvaluation], PrivateStateAttr]
    """Accumulated grader evaluations across rubrics."""

    _current_grading_run_id: Annotated[str, PrivateStateAttr]
    """Tracking id for the active grading run."""

    _active_rubric: Annotated[str, PrivateStateAttr]
    """The rubric that minted ``_current_grading_run_id``."""
