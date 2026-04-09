"""State schema and reducers for team coordination."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def _merge_team_members(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge team members by name — latest update wins per member."""
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for member in left or []:
        n = member.get("name", "")
        if n:
            by_name[n] = dict(member)
            order.append(n)
    for member in right or []:
        n = member.get("name", "")
        if not n:
            continue
        if n in by_name:
            by_name[n].update(member)
        else:
            by_name[n] = dict(member)
            order.append(n)
    return [by_name[n] for n in order if n in by_name]


class TeamState(TypedDict, total=False):
    """State mixin for team coordination."""

    team_members: Annotated[list[dict[str, Any]], _merge_team_members]
    team_name: str | None
