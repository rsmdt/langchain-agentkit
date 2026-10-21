from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_agentkit.extensions.teams.bus import TeamMessage, TeamMessageBus, teammate_loop
from langchain_agentkit.extensions.teams.llm_filter import is_team_tagged
from langchain_agentkit.extensions.teams.task_router import classify_and_process


@pytest.mark.asyncio
async def test_teammate_message_to_lead_names_its_sender():
    bus = TeamMessageBus()
    bus.register("lead")
    raw = TeamMessage(id="m1", sender="alice", receiver="lead", content="done", timestamp=1.0)

    update = await classify_and_process([raw], [], bus)

    [message] = update["messages"]
    assert isinstance(message, HumanMessage)
    assert not is_team_tagged(message)
    assert message.name == "alice"
    assert "alice" in message.content
    assert message.content.endswith("done")
    assert message.content != "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", ["lead", "bob"])
async def test_instruction_reaching_a_teammate_names_its_sender(sender: str):
    bus = TeamMessageBus()
    for name in ("r1", "lead", "bob"):
        bus.register(name)
    seen: list[list[Any]] = []

    async def _ainvoke(state: dict[str, Any]) -> dict[str, Any]:
        seen.append(list(state["messages"]))
        return {"messages": [*state["messages"], AIMessage(content="ok")]}

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=_ainvoke)
    loop = asyncio.create_task(teammate_loop("r1", graph, bus, initial_history=[]))

    await bus.send(sender, "r1", "do it")
    await bus.receive(sender, timeout=2.0)
    await bus.send("lead", "r1", '{"type":"shutdown_request"}')
    await asyncio.wait_for(loop, timeout=2.0)

    incoming = seen[0][-1]
    assert isinstance(incoming, HumanMessage)
    assert incoming.name == sender
    assert sender in incoming.content
    assert incoming.content.endswith("do it")
