# ruff: noqa: N801, N805
"""Shared workspace — SkillsExtension + FilesystemExtension on the same directory.

SkillsExtension discovers skills from a directory on disk.
FilesystemExtension gives the agent Read/Write/Edit/Glob/Grep on a workspace.

Both operate on the real OS filesystem — no virtual layer needed.
"""

import asyncio
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    FilesystemExtension,
    SkillsExtension,
    agent,
)

# Pre-populate a workspace directory
workspace = Path(tempfile.mkdtemp(prefix="agentkit_"))

(workspace / "brief.md").write_text("""\
# Project Brief
Client: Acme Corp
Objective: Estimate TAM for industrial IoT sensors in Europe
Deadline: 2024-Q4
""")

(workspace / "notes.md").write_text("""\
# Research Notes
- European IoT market growing ~20% YoY
- Industrial segment is 35% of total IoT
- Key players: Siemens, Bosch, ABB
""")

# Skills directory (relative to project root)
skills_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "skills"


class researcher(agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [
        # Skills discovered from real directory — provides only Skill tool
        SkillsExtension(skills=skills_dir),
        # Filesystem tools on the workspace — provides Read, Write, Edit, Glob, Grep, LS, MultiEdit
        FilesystemExtension(root=workspace),
    ]
    prompt = """\
You are a research assistant. Use the Skill tool to load methodologies,
and Read/Glob to access workspace files. Write your findings to the workspace."""

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}


async def main():
    graph = researcher.compile()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            "Read the project brief, load the market-sizing skill, "
            "then write an initial analysis to /analysis.md"
        )]}
    )

    print(result["messages"][-1].content[:500])

    # Check what the agent wrote
    analysis_path = workspace / "analysis.md"
    if analysis_path.exists():
        print(f"\n--- Agent wrote to {analysis_path} ---\n{analysis_path.read_text()[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
