"""Composable state schemas for LangGraph agents.

``AgentKitState`` is the minimal base — just messages, sender, and jump_to.
Extensions add state keys via mixins (e.g., ``TasksState``, ``TeamState``).

Usage::

    from langchain_agentkit.composition.state import AgentKitState
    from langchain_agentkit.extensions.tasks.state import TasksState

    # Compose manually
    class MyState(AgentKitState, TasksState):
        my_field: str

    # Or let AgentKit compose from extensions automatically
    kit = AgentKit(extensions=[TasksExtension()])
    kit.state_schema  # → composed TypedDict with messages + tasks
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, cast, get_args, get_type_hints, override

from langgraph.graph.message import add_messages


def _last_writer_wins(left: str | None, right: str | None) -> str | None:
    """Reducer that takes the most recent value (right wins)."""
    return right


class _PrivateStateAttr:
    """Marker type for :data:`PrivateStateAttr`. Use the singleton instance.

    Deliberately **not callable**: LangGraph treats any callable in a key's
    ``Annotated`` metadata as a reducer, so the marker must be a plain
    sentinel value to be ignored by channel construction. Place it before any
    reducer in the metadata (``Annotated[T, PrivateStateAttr, reducer]``) so
    LangGraph still picks the reducer up as the last item.
    """

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "PrivateStateAttr"


PrivateStateAttr = _PrivateStateAttr()
"""Annotation marker that hides a state key from the public I/O schema.

Annotate an internal bookkeeping key with ``Annotated[T, PrivateStateAttr]``
to keep it out of a graph's **input and output** schemas. The key remains a
real channel — nodes, hooks, reducers, and ``get_state()`` all see it — but
callers cannot set it on invocation input and it does not appear in the
``invoke`` result.

Observe such keys via ``get_state(config).values`` on a checkpointed thread,
or via whatever callbacks / stream events the owning extension exposes. Use it
for state an extension manages entirely on its own behalf, where surfacing the
raw key in the public shape would be noise (or an invitation to set it
incorrectly).
"""


def _has_private_marker(annotation: Any) -> bool:
    """True if ``annotation`` (possibly wrapped) carries :class:`PrivateStateAttr`.

    Walks ``Annotated`` metadata and descends through wrappers such as
    ``NotRequired[Annotated[T, PrivateStateAttr]]``.
    """
    if PrivateStateAttr in getattr(annotation, "__metadata__", ()):
        return True
    return any(_has_private_marker(arg) for arg in get_args(annotation))


def public_state_schema(state_schema: type) -> type | None:
    """Build a TypedDict mirroring ``state_schema`` minus private keys.

    Returns a new ``total=False`` TypedDict containing every key *not*
    annotated with :class:`PrivateStateAttr`, preserving each kept key's
    annotation (so reducers like ``add_messages`` survive). Returns ``None``
    when nothing is private — the caller can then use the full schema for
    I/O unchanged, so non-private graphs are untouched.
    """
    hints = get_type_hints(state_schema, include_extras=True)
    public = {name: typ for name, typ in hints.items() if not _has_private_marker(typ)}
    if len(public) == len(hints):
        return None
    return cast("type", TypedDict("PublicState", public, total=False))  # type: ignore[operator]


class AgentKitState(TypedDict, total=False):
    """Minimal base state — always present in any agentkit graph.

    Contains only the fields required for the ReAct loop to function.
    Extension adds additional keys via mixin TypedDicts.
    """

    messages: Annotated[list[Any], add_messages]
    sender: str
    _agentkit_jump_to: Annotated[str | None, _last_writer_wins]
