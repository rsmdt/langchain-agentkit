"""Review cadence policy for :class:`RubricExtension`.

A single value object that answers *when* the grader checks the work:

- ``min`` (turns)   — don't review before this many assistant turns since the
  last review (give the agent room to work).
- ``max`` (turns)   — force a review once this many turns have passed; the
  agent may call ``request_review()`` to be graded earlier, anywhere in
  ``[min, max]``.
- ``stop`` (reviews) — give up after this many reviews without a ``satisfied``
  verdict (the graceful terminal, distinct from the hard per-invoke
  ``max_turns`` ceiling).

``min``/``max`` are turns; ``stop`` is a review count — they pace the checks
and cap the checks respectively, which is why they carry different units.
"""

from __future__ import annotations

from dataclasses import dataclass

HARD_CAP = 20
"""Upper bound for any single field — a runaway-config backstop."""


@dataclass(frozen=True)
class ReviewPolicy:
    """When the grader reviews the agent's work. See module docstring."""

    min: int = 1  # noqa: A003 — domain field name; not shadowing the builtin in use
    max: int = 1  # noqa: A003
    stop: int = 5

    def __post_init__(self) -> None:
        for name, value in (("min", self.min), ("max", self.max), ("stop", self.stop)):
            if not isinstance(value, int) or isinstance(value, bool):
                msg = f"ReviewPolicy.{name} must be an int, got {type(value).__name__}."
                raise TypeError(msg)
        if not 1 <= self.min <= self.max <= HARD_CAP:
            msg = (
                f"ReviewPolicy requires 1 <= min <= max <= {HARD_CAP}, "
                f"got min={self.min}, max={self.max}."
            )
            raise ValueError(msg)
        if not 1 <= self.stop <= HARD_CAP:
            msg = f"ReviewPolicy.stop must be in [1, {HARD_CAP}], got {self.stop}."
            raise ValueError(msg)

    @property
    def has_window(self) -> bool:
        """True when the agent has discretion over *when* to be reviewed.

        With ``min == max`` there is no window — grading is fixed-cadence, so
        ``request_review()`` would be meaningless and the tool is not offered.
        """
        return self.max > self.min

    @classmethod
    def coerce(cls, value: int | dict[str, int] | ReviewPolicy) -> ReviewPolicy:
        """Build a policy from an int, a dict, or a policy.

        - ``ReviewPolicy`` — returned as-is.
        - ``int N`` — ``ReviewPolicy(min=1, max=N)`` ("grade by turn N; agent
          may request earlier"). ``1`` means every turn.
        - ``dict`` — keyword fields (``{"min": .., "max": .., "stop": ..}``).
        """
        if isinstance(value, ReviewPolicy):
            return value
        if isinstance(value, bool):
            msg = "ReviewPolicy: review must be an int, dict, or ReviewPolicy, not bool."
            raise TypeError(msg)
        if isinstance(value, int):
            return cls(min=1, max=value)
        if isinstance(value, dict):
            return cls(**value)
        msg = (
            "ReviewPolicy: review must be an int, dict, or ReviewPolicy, "
            f"got {type(value).__name__}."
        )
        raise TypeError(msg)
