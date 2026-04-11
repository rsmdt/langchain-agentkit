# ruff: noqa: N805
"""Daytona sandbox — full integration with filesystem, skills, and agents.

Demonstrates using a Daytona sandbox as the backend for:
- FilesystemExtension — file tools (Read, Write, Edit, Glob, Grep, Bash)
- SkillsExtension — skills discovered from sandbox filesystem
- AgentsExtension — agent definitions discovered from sandbox filesystem

Bash requires human approval via the DEFAULT_RULESET permission preset.

Prerequisites:
    pip install daytona-sdk langchain-openai

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
import sys
from pathlib import Path

from daytona_sdk import Daytona, DaytonaConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    Agent,
    AgentsExtension,
    FilesystemExtension,
    SkillsExtension,
)
from langchain_agentkit.backends import DaytonaBackend
from langchain_agentkit.permissions import DEFAULT_RULESET

# Paths inside the sandbox where skills and agents are uploaded
SKILLS_PATH = ".agentkit/skills"
AGENTS_PATH = ".agentkit/agents"


def seed_sandbox(backend: DaytonaBackend) -> None:
    """Upload local skills/ and agents/ directories into the sandbox."""
    local_dir = Path(__file__).parent

    for local_file in sorted((local_dir / "skills").rglob("*")):
        if local_file.is_file():
            rel = local_file.relative_to(local_dir / "skills")
            backend.write(f"{SKILLS_PATH}/{rel}", local_file.read_text())
            print(f"  uploaded {SKILLS_PATH}/{rel}")

    for local_file in sorted((local_dir / "agents").rglob("*")):
        if local_file.is_file():
            rel = local_file.relative_to(local_dir / "agents")
            backend.write(f"{AGENTS_PATH}/{rel}", local_file.read_text())
            print(f"  uploaded {AGENTS_PATH}/{rel}")


def build_agent(backend: DaytonaBackend):
    """Build the agent with Daytona-backed extensions."""

    class SandboxAgent(Agent):
        model = ChatOpenAI(model="gpt-4o")
        extensions = [
            SkillsExtension(skills=SKILLS_PATH, backend=backend),
            AgentsExtension(agents=AGENTS_PATH, backend=backend),
            FilesystemExtension(backend=backend, permissions=DEFAULT_RULESET),
        ]
        prompt = """\
You are a development assistant working in a sandboxed environment.
You have access to a filesystem, skills, and specialist agents.
Use your tools to explore, read, and modify files."""

        async def handler(state, *, llm, tools, prompt):
            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.bind_tools(tools).ainvoke(messages)
            return {"messages": [response]}

    return SandboxAgent


async def main() -> None:
    config = DaytonaConfig(
        api_key=os.environ.get("DAYTONA_API_KEY"),
        api_url=os.environ.get("DAYTONA_API_URL"),
    )
    daytona = Daytona(config=config)

    print("Creating Daytona sandbox...")
    sandbox = daytona.create()
    print(f"  sandbox id: {sandbox.id}")
    print(f"  work dir:   {sandbox.get_work_dir()}")

    try:
        backend = DaytonaBackend(sandbox)

        print("\nSeeding sandbox filesystem...")
        seed_sandbox(backend)

        print("\nBuilding agent...")
        agent_class = build_agent(backend)
        graph = agent_class().compile()

        print("Running agent...\n")
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage("List all files in the workspace, then summarize what you find.")
                ]
            }
        )

        print("=" * 60)
        print("AGENT RESPONSE")
        print("=" * 60)
        print(result["messages"][-1].content)

    finally:
        print("\nCleaning up sandbox...")
        sandbox.delete()
        print("Done.")


if __name__ == "__main__":
    if not os.environ.get("DAYTONA_API_KEY"):
        print("Error: DAYTONA_API_KEY environment variable is required.")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    asyncio.run(main())
