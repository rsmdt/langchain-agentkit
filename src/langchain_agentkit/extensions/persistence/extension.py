"""MessagePersistenceExtension — persist each turn's generated messages.

Captures the messages produced during a single ``ainvoke`` (one turn)
and forwards them to a caller-supplied async callback.  Uses only
LangGraph-standard state and config:

- ``before_run`` snapshots the IDs of messages already in state.
- ``after_run`` diffs the final message list against that snapshot and
  forwards the newly generated messages to the callback.

The only config key read is LangGraph's standard ``thread_id``.  No
application-specific coordination is required from the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langchain_agentkit.extension import Extension
from langchain_agentkit.hooks import after, before

logger = logging.getLogger(__name__)


PersistCallback = Callable[..., Awaitable[None]]


class _PersistState(TypedDict, total=False):
    """State mixin tracking which message IDs existed at turn start."""

    _persist_seen_ids: list[str]


class MessagePersistenceExtension(Extension):
    """Persist messages generated during each turn to an external sink.

    The caller supplies an async ``persist`` callable.  It is invoked
    once per ``ainvoke`` (turn) with the messages that were added to
    graph state during that turn.

    Example::

        async def write_to_db(*, thread_id, messages):
            async with session_factory() as session:
                session.add_all([serialize(m, thread_id) for m in messages])
                await session.commit()

        class MyAgent(Agent):
            model = ChatOpenAI(model="gpt-4o")
            extensions = [MessagePersistenceExtension(persist=write_to_db)]

    Turn delta is computed by ID: ``before_run`` snapshots the set of
    message IDs already in state; ``after_run`` filters the final list
    to messages whose ID was not in that snapshot.  Messages without an
    ``id`` attribute are skipped (all provider-returned LangChain
    messages carry IDs).

    Persistence failures are caught and logged — a failed DB write
    never fails the turn.  If the caller wants fail-fast semantics they
    can raise inside their callback and wrap ``ainvoke`` themselves.
    """

    def __init__(self, *, persist: PersistCallback) -> None:
        self._persist = persist

    @property
    def state_schema(self) -> type:
        return _PersistState

    @before("run")
    async def _snapshot_seen_ids(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any]:
        seen: list[str] = []
        for msg in state.get("messages") or []:
            mid = getattr(msg, "id", None)
            if mid:
                seen.append(mid)
        return {"_persist_seen_ids": seen}

    @after("run")
    async def _persist_generated(
        self,
        *,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        seen = set(state.get("_persist_seen_ids") or [])
        generated: list[Any] = []
        for msg in state.get("messages") or []:
            mid = getattr(msg, "id", None)
            if mid and mid not in seen:
                generated.append(msg)

        if not generated:
            return None

        thread_id: str | None = None
        if runtime and getattr(runtime, "config", None):
            thread_id = (runtime.config.get("configurable") or {}).get("thread_id")

        try:
            await self._persist(thread_id=thread_id, messages=generated)
        except Exception:
            logger.exception(
                "MessagePersistenceExtension: persist callback failed "
                "(thread_id=%s, message_count=%d)",
                thread_id,
                len(generated),
            )
        return None
