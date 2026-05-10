# ruff: noqa: N805
"""Mirage-backed agent — multi-mount unified VFS with shell access.

Mirage exposes multiple resources (RAM, Disk, S3, Slack, GitHub, …) under
one filesystem; ``MirageBackend`` implements ``FilesystemProtocol`` and
``SandboxProtocol`` over a pre-constructed ``mirage.Workspace``. This
example demonstrates the multi-mount pattern with two local resources:

- ``/.agentkit`` — host directory mounted via ``DiskResource`` (read-only).
  No ``read_tree()`` upload needed — the agent reads config directly from
  disk through Mirage's filesystem abstraction.
- ``/workspace`` — in-memory ``RAMResource`` for the agent's working
  files (writable).

The agent has shell access via Mirage's tree-sitter-bash interpreter
and can pipe across mounts: ``cat /.agentkit/AGENTS.md | wc -l`` reads
disk and pipes through the in-process ``wc`` command.

Prerequisites:
    pip install mirage-ai langchain-openai

Environment:
    OPENAI_API_KEY — OpenAI API key

Usage:
    python examples/mirage/agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from mirage import DiskResource, MountMode, RAMResource, Workspace

from langchain_agentkit import (
    Agent,
    AgentsExtension,
    FilesystemExtension,
    SkillsExtension,
)
from langchain_agentkit.backends.mirage import MirageBackend

ROOT = "/.agentkit"
SKILLS = f"{ROOT}/skills"
AGENTS = f"{ROOT}/agents"
PROMPT = f"{ROOT}/AGENTS.md"

WORKSPACE = "/workspace"


class CodingAgent(Agent):
    model = ChatOpenAI(model="gpt-4o")

    async def prompt(self):
        return (await self.backend.read(PROMPT)).content

    def extensions(self):
        return [
            SkillsExtension(skills=SKILLS, backend=self.backend),
            AgentsExtension(agents=AGENTS, backend=self.backend),
            FilesystemExtension(backend=self.backend),
        ]

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


async def main() -> None:
    host_agentkit = Path(__file__).parent / ".agentkit"

    ws = Workspace(
        {
            ROOT: (DiskResource(root=str(host_agentkit)), MountMode.READ),
            WORKSPACE: (RAMResource(), MountMode.WRITE),
        }
    )

    try:
        backend = MirageBackend(ws)

        graph = await CodingAgent(backend=backend).compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "List all files in the workspace, then summarize what you find. "
                        f"Write the summary to {WORKSPACE}/summary.md."
                    )
                ]
            }
        )

        print(result["messages"][-1].content)
    finally:
        await ws.close()


if __name__ == "__main__":
    # Requires OPENAI_API_KEY in the environment.
    asyncio.run(main())
