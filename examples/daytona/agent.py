# ruff: noqa: N805
"""Daytona sandbox — full integration with filesystem, skills, and agents.

The agent harness — system prompt, skills, and specialist agents —
lives entirely in the sandbox under ``/.agentkit/``. The host-side
``.agentkit/`` directory is a 1:1 mirror of what the agent sees;
``seed_directory`` copies the tree on startup. Bash requires human
approval via the DEFAULT_RULESET permission preset.

Prerequisites:
    pip install daytona langchain-openai

Environment:
    DAYTONA_API_KEY  — Daytona API key
    DAYTONA_API_URL  — Daytona API URL (optional)
    OPENAI_API_KEY   — OpenAI API key

Usage:
    python examples/daytona/agent.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from daytona import Daytona, DaytonaConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    Agent,
    AgentsExtension,
    FilesystemExtension,
    SkillsExtension,
)
from langchain_agentkit.backends import read_tree
from langchain_agentkit.backends.daytona import DaytonaBackend
from langchain_agentkit.permissions import DEFAULT_RULESET

ROOT = "/.agentkit"
SKILLS = f"{ROOT}/skills"
AGENTS = f"{ROOT}/agents"
PROMPT = f"{ROOT}/AGENTS.md"


class CodingAgent(Agent):
    model = ChatOpenAI(model="gpt-4o")

    async def prompt(self):
        return (await self.backend.read(PROMPT)).content

    def extensions(self):
        return [
            SkillsExtension(skills=SKILLS, backend=self.backend),
            AgentsExtension(agents=AGENTS, backend=self.backend),
            FilesystemExtension(backend=self.backend, permissions=DEFAULT_RULESET),
        ]

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


async def main() -> None:
    config = DaytonaConfig(
        api_key=os.environ.get("DAYTONA_API_KEY"),
        api_url=os.environ.get("DAYTONA_API_URL"),
    )
    sandbox = Daytona(config=config).create()

    try:
        backend = DaytonaBackend(sandbox)
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
        sandbox.delete()


if __name__ == "__main__":
    # Requires DAYTONA_API_KEY and OPENAI_API_KEY in the environment.
    asyncio.run(main())
