# ruff: noqa: N805
"""AgentFS-backed agent — file-only backend, no bash tool.

AgentFS is a SQLite-backed virtual filesystem; ``AgentFSBackend``
implements ``FilesystemProtocol`` only (no ``SandboxProtocol``, so no
``Bash`` tool is registered). Agent state —
including the system prompt at ``/.agentkit/AGENTS.md`` — persists in
``./agent-state.db``.

Prerequisites:
    pip install agentfs-sdk langchain-openai

Environment:
    OPENAI_API_KEY — OpenAI API key

Usage:
    python examples/agentfs/agent.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agentfs_sdk import AgentFS, AgentFSOptions
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    Agent,
    AgentsExtension,
    FilesystemExtension,
    SkillsExtension,
)
from langchain_agentkit.backends import read_tree
from langchain_agentkit.backends.agentfs import AgentFSBackend

ROOT = "/.agentkit"
SKILLS = f"{ROOT}/skills"
AGENTS = f"{ROOT}/agents"
PROMPT = f"{ROOT}/AGENTS.md"

DB = "./agent-state.db"


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
    fs = await AgentFS.open(AgentFSOptions(path=DB))

    try:
        backend = AgentFSBackend(fs)
        await backend.upload(read_tree(Path(__file__).parent / ".agentkit", ROOT))

        graph = await CodingAgent(backend=backend).compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage("List all files in the workspace, then summarize what you find.")
                ]
            }
        )

        print(result["messages"][-1].content)
    finally:
        await fs.close()


if __name__ == "__main__":
    # Requires OPENAI_API_KEY in the environment.
    asyncio.run(main())
