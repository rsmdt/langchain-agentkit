"""TeamMessageBus and teammate execution loop."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

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


@dataclass
class ActiveTeam:
    """Runtime state for an active team."""

    name: str
    bus: TeamMessageBus
    members: dict[str, asyncio.Task[str]]
    member_types: dict[str, str]


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


async def _teammate_loop(
    member_name: str,
    compiled_graph: Any,
    message_bus: TeamMessageBus,
    thread_id: str | None = None,
) -> str:
    """Event loop for a single teammate."""
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    while True:
        msg = await message_bus.receive(member_name, timeout=30.0)
        if msg is None:
            continue
        if _is_shutdown_request(msg.content) and msg.sender == "lead":
            # Acknowledge shutdown — only accept from lead
            response = json.dumps({"type": "shutdown_response", "approve": True})
            await message_bus.send(member_name, msg.sender, response)
            return "shutdown"

        try:
            result = await compiled_graph.ainvoke(
                {
                    "messages": [HumanMessage(content=msg.content)],
                    "sender": member_name,
                },
                config=config,
            )
            final = result["messages"][-1].content if result.get("messages") else "No response"
        except Exception as exc:
            final = f"Error during execution: {exc}"

        await message_bus.send(member_name, msg.sender, final)
