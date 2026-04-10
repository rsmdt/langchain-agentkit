"""State schema and reducers for team coordination.

The ``team`` channel stores the metadata required to rebuild a team on
any process — team name, created_at, and a list of ``TeammateSpec``
entries.  Teammate runtime (bus, asyncio.Tasks, compiled graphs) is NOT
persisted; it is reconstructed from this metadata at turn start.

Teammate conversation history is NOT stored here either — it lives in
the shared ``state["messages"]`` channel, tagged per teammate via
``additional_kwargs["team"]["member"]``.  See ``teams/filter.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict


class TeammateSpec(TypedDict):
    """Serializable description of a teammate sufficient to rebuild its graph.

    Always accessed via dict notation: ``spec["member_name"]``.  Never
    attribute access.
    """

    member_name: str
    kind: Literal["predefined", "dynamic"]
    agent_id: NotRequired[str]
    """Required when kind == 'predefined'.  References a name in the agents roster."""
    system_prompt: NotRequired[str]
    """Required when kind == 'dynamic'.  Stored WITHOUT the teammate addendum.

    The addendum is re-applied at rehydration time so it can evolve across
    deploys without stale copies surviving in old state.
    """
    allowed_tools: NotRequired[list[str]]
    """Optional tool whitelist (forward-compat hook; currently unused)."""


class TeamMetadata(TypedDict, total=False):
    """Top-level team metadata stored in ``state["team"]``."""

    name: str
    members: list[TeammateSpec]
    created_at: str  # ISO timestamp for debugging


def _team_reducer(
    left: TeamMetadata | None,
    right: TeamMetadata | None,
) -> TeamMetadata | None:
    """Replace-wins — including explicit ``None`` (dissolve).

    LangGraph only invokes a reducer when a node has actually returned
    an update for this channel; absence of the key short-circuits the
    reducer entirely.  So if the reducer IS called with ``right=None``,
    that's an explicit clear (typically from ``TeamDissolve``) and must
    win — never fall back to ``left``.

    Concurrent ``TeamCreate`` + ``TeamDissolve`` within a single
    superstep is serialized by ``TeamExtension._team_lock`` at the
    runtime level; whichever tool body successfully returns a Command
    wins.  If both return Commands (e.g. dissolve-then-create in the
    same parallel tool batch), LangGraph merges them in order and the
    last write wins — document this as user error and let the lock
    catch 99% of cases.
    """
    return right


class TeamState(TypedDict, total=False):
    """State mixin for team coordination."""

    team: Annotated[TeamMetadata | None, _team_reducer]
