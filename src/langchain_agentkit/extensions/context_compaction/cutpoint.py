"""Cutpoint detection — pick where to bisect the history for compaction.

Algorithm: walk backward from the newest message accumulating estimated
tokens. Stop once we've retained at least ``keep_recent_tokens``. The cut
must land at a ``HumanMessage`` or ``AIMessage`` boundary — never inside
a tool call/tool-result pair, which would orphan the pairing and
confuse provider tool-use semantics.

When the chosen boundary is an assistant message (i.e. the cut splits a
turn), :attr:`CutPoint.turn_start` points back to the user message that
opened that turn so the summarizer can render a separate "turn prefix"
summary alongside the main history summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extensions.context_compaction.token_accounting import (
    estimate_tokens,
    is_assistant_like,
    is_user_like,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class CutPoint:
    """Result of :func:`find_cut_point`.

    ``first_kept_index`` — index of the first message preserved verbatim.
    ``turn_start_index`` — when the cut splits a turn, the index of the
    user message that started it; ``-1`` otherwise.
    ``is_split_turn`` — ``True`` when the cut landed mid-turn.
    """

    first_kept_index: int
    turn_start_index: int
    is_split_turn: bool


def _find_valid_cut_indices(
    messages: Sequence[Any],
    start: int,
    end: int,
) -> list[int]:
    """Return indices where cutting is safe (user / assistant boundaries)."""
    out: list[int] = []
    for i in range(start, end):
        msg = messages[i]
        if is_user_like(msg) or is_assistant_like(msg):
            out.append(i)
    return out


def find_turn_start_index(
    messages: Sequence[Any],
    cut_index: int,
    start: int,
) -> int:
    """Walk backward from ``cut_index`` to the user message opening this turn."""
    for i in range(cut_index, start - 1, -1):
        if is_user_like(messages[i]):
            return i
    return -1


def find_cut_point(
    messages: Sequence[Any],
    start: int,
    end: int,
    keep_recent_tokens: int,
) -> CutPoint:
    """Pick the cut point keeping at least ``keep_recent_tokens`` of tail.

    Walks backward, accumulating per-message token estimates. Once the
    accumulator meets the budget, snaps to the nearest valid cut at or
    after the current index. Returns a :class:`CutPoint` describing where
    to bisect and whether the cut splits a turn.

    If no valid cut exists in the window, returns a :class:`CutPoint`
    with ``first_kept_index == start`` so callers no-op safely.
    """
    valid = _find_valid_cut_indices(messages, start, end)
    if not valid:
        return CutPoint(first_kept_index=start, turn_start_index=-1, is_split_turn=False)

    accumulated = 0
    cut_index = valid[0]

    for i in range(end - 1, start - 1, -1):
        accumulated += estimate_tokens(messages[i])
        if accumulated >= keep_recent_tokens:
            # Find the closest valid cut at or after this index.
            for c in valid:
                if c >= i:
                    cut_index = c
                    break
            break
    else:
        # Never hit the budget — nothing to drop. Keep the newest valid cut
        # (last entry in ``valid``) so the caller can no-op.
        cut_index = valid[-1] if valid else start

    cut_msg = messages[cut_index]
    is_user_cut = is_user_like(cut_msg)
    turn_start = -1 if is_user_cut else find_turn_start_index(messages, cut_index, start)
    return CutPoint(
        first_kept_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=not is_user_cut and turn_start != -1,
    )
