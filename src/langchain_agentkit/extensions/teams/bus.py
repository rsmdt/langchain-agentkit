"""TeamMessageBus and teammate execution loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.extensions.teams.filter import tag_message

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

_logger = logging.getLogger("langchain_agentkit.extensions.teams")

# Sentinel content string used to signal teammate shutdown via the message bus.
SHUTDOWN_SIGNAL = "__shutdown__"


@dataclass
class TeamMessage:
    """A single message passed between team members via the bus."""

    id: str
    sender: str
    receiver: str
    content: str
    timestamp: float


class TeamMessageBus:
    """asyncio.Queue-based message bus for team coordination."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[TeamMessage]] = {}

    def register(self, agent_name: str) -> None:
        if agent_name in self._queues:
            return
        self._queues[agent_name] = asyncio.Queue()

    def unregister(self, agent_name: str) -> None:
        self._queues.pop(agent_name, None)

    async def send(self, from_agent: str, to_agent: str, content: str) -> None:
        queue = self._queues.get(to_agent)
        if queue is None:
            raise ValueError(f"Agent '{to_agent}' is not registered on the bus.")
        message = TeamMessage(
            id=str(uuid.uuid4()),
            sender=from_agent,
            receiver=to_agent,
            content=content,
            timestamp=time.time(),
        )
        await queue.put(message)

    async def broadcast(self, from_agent: str, content: str) -> None:
        for name in self._queues:
            if name != from_agent:
                await self.send(from_agent, name, content)

    async def receive(self, agent_name: str, timeout: float = 5.0) -> TeamMessage | None:
        queue = self._queues.get(agent_name)
        if queue is None:
            return None
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return None

    def pending_count(self, agent_name: str) -> int:
        queue = self._queues.get(agent_name)
        if queue is None:
            return 0
        return queue.qsize()

    async def request_response(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        *,
        request_id: str | None = None,
        timeout: float = 10.0,
    ) -> TeamMessage | None:
        """Send a message and wait for a response with a matching ``request_id``.

        The caller embeds ``request_id`` in its JSON payload. The responder
        must include the same ``request_id`` in the reply. Non-matching
        messages that arrive while waiting are re-queued so they are not lost.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        await self.send(from_agent, to_agent, content)

        deadline = asyncio.get_event_loop().time() + timeout
        stashed: list[TeamMessage] = []

        try:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None

                msg = await self.receive(from_agent, timeout=remaining)
                if msg is None:
                    return None

                # Check if this is our response
                try:
                    parsed = json.loads(msg.content)
                    if isinstance(parsed, dict) and parsed.get("request_id") == request_id:
                        return msg
                except (json.JSONDecodeError, TypeError):
                    pass

                # Not our response — stash it for later
                stashed.append(msg)
        finally:
            # Re-queue stashed messages so they aren't lost
            queue = self._queues.get(from_agent)
            if queue is not None:
                for m in stashed:
                    await queue.put(m)


@dataclass
class ActiveTeam:
    """Runtime state for an active team."""

    name: str
    bus: TeamMessageBus
    members: dict[str, asyncio.Task[str]]
    member_types: dict[str, str]


def task_status(task: asyncio.Task[str]) -> str:
    """Classify an ``asyncio.Task`` as running / completed / cancelled / failed."""
    if not task.done():
        return "running"
    try:
        task.result()
        return "completed"
    except asyncio.CancelledError:
        return "cancelled"
    except Exception:
        return "failed"


def _is_shutdown_request(content: str) -> bool:
    """Check if a message content is a shutdown request.

    Supports both the legacy SHUTDOWN_SIGNAL string and the new
    structured ``{"type": "shutdown_request"}`` format.
    """
    if content == SHUTDOWN_SIGNAL:
        return True
    try:
        parsed = json.loads(content)
        return isinstance(parsed, dict) and parsed.get("type") == "shutdown_request"
    except (json.JSONDecodeError, TypeError):
        return False


async def _teammate_loop(  # noqa: C901
    member_name: str,
    compiled_graph: Any,
    message_bus: TeamMessageBus,
    initial_history: list[BaseMessage] | None = None,
    capture_buffer: list[BaseMessage] | None = None,
) -> str:
    """Event loop for a single teammate, with message capture.

    The loop keeps a local ``history`` list that starts from
    ``initial_history`` (the filtered slice from state at rehydration
    time, or ``[]`` for a fresh team) and grows across bus messages in
    one turn.  Each new message the teammate graph produces is tagged
    with ``additional_kwargs["team"]["member"]=<member_name>`` and appended to ``capture_buffer``
    so the lead's hook can flush them into persisted ``state["messages"]``.

    PRECONDITION: the teammate's compiled graph must use an append-style
    reducer on ``messages`` (e.g. ``add_messages``).  Graphs built via
    ``build_ephemeral_graph`` / ``_compile_with_proxy_tasks`` satisfy
    this.  A non-append reducer would break the ``result[len(history):]``
    slice used to isolate newly-produced messages; the assertion below
    fires immediately with a clear error in that case.

    ``capture_buffer`` and ``initial_history`` default to ``None`` only
    so legacy call sites do not have to be updated in one atomic change;
    in practice they should always be passed.
    """
    history: list[BaseMessage] = list(initial_history or [])
    buffer: list[BaseMessage] | None = capture_buffer

    while True:
        msg = await message_bus.receive(member_name, timeout=30.0)
        if msg is None:
            continue
        if _is_shutdown_request(msg.content) and msg.sender == "lead":
            # Acknowledge shutdown — only accept from lead
            response = json.dumps({"type": "shutdown_response", "approve": True})
            await message_bus.send(member_name, msg.sender, response)
            return "shutdown"

        # Construct the teammate's view of the incoming instruction and
        # append it to both the local history and the shared capture
        # buffer (so it lands in persisted state after the next flush).
        incoming = HumanMessage(content=msg.content)
        tag_message(incoming, member_name)
        history.append(incoming)
        if buffer is not None:
            buffer.append(incoming)

        # Invoke the teammate's graph with the full local history.
        # No checkpointer config — teammates are stateless across
        # invocations; history persistence lives in state["messages"].
        try:
            result = await compiled_graph.ainvoke(
                {"messages": history, "sender": member_name},
            )
        except Exception as exc:  # noqa: BLE001
            _logger.exception(
                "Teammate %r invocation raised; capturing error message.",
                member_name,
            )
            error_msg = AIMessage(content=f"Error during execution: {exc}")
            tag_message(error_msg, member_name)
            history.append(error_msg)
            if buffer is not None:
                buffer.append(error_msg)
            await message_bus.send(member_name, msg.sender, str(error_msg.content))
            continue

        result_messages: list[BaseMessage] = list(result.get("messages") or [])
        if len(result_messages) < len(history):
            raise AssertionError(  # noqa: TRY003
                f"Teammate {member_name!r} graph returned fewer messages "
                f"({len(result_messages)}) than input ({len(history)}). "
                "This indicates a non-append reducer on the messages "
                "channel — see _teammate_loop precondition."
            )

        new_messages = result_messages[len(history) :]
        for m in new_messages:
            tag_message(m, member_name)
        history.extend(new_messages)
        if buffer is not None:
            buffer.extend(new_messages)

        # Send the final reply on the bus back to the caller.
        if new_messages and hasattr(new_messages[-1], "content"):
            final_content = str(new_messages[-1].content)
        else:
            final_content = "(no response)"
        await message_bus.send(member_name, msg.sender, final_content)
