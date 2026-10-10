from __future__ import annotations

from typing import Annotated, TypedDict


def _append_skill_names(left: list[str] | None, right: list[str] | None) -> list[str]:
    return [*(left or []), *(right or [])]


class SkillsState(TypedDict, total=False):
    pinned_skills: Annotated[list[str], _append_skill_names]
