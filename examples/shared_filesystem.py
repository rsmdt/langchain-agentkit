# ruff: noqa: N801, N805
"""Shared VFS — SkillsExtension + FilesystemExtension on the same filesystem.

When you pass a VirtualFilesystem to SkillsExtension, it populates the
VFS with skill files but does NOT include filesystem tools. Add
FilesystemExtension with the same VFS to get Read/Write/Edit/Glob/Grep.

This gives you one unified filesystem where skills coexist with your
own data — the agent can Glob across both.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    FilesystemExtension,
    SkillsExtension,
    VirtualFilesystem,
    agent,
)

# Shared VFS — skills and workspace data in one filesystem
vfs = VirtualFilesystem()

# Pre-populate with workspace data
vfs.write("/workspace/brief.md", """\
# Project Brief
Client: Acme Corp
Objective: Estimate TAM for industrial IoT sensors in Europe
Deadline: 2024-Q4
""")

vfs.write("/workspace/notes.md", """\
# Research Notes
- European IoT market growing ~20% YoY
- Industrial segment is 35% of total IoT
- Key players: Siemens, Bosch, ABB
""")


class researcher(agent):
    llm = ChatOpenAI(model="gpt-4o")
    extensions = [
        # Skills loaded into VFS at /skills/ — provides only Skill tool
        SkillsExtension(skills="skills/", filesystem=vfs),
        # Filesystem tools on the same VFS — provides Read, Write, Edit, Glob, Grep
        FilesystemExtension(filesystem=vfs),
    ]
    prompt = """\
You are a research assistant. Use the Skill tool to load methodologies,
and Read/Glob to access workspace files. Write your findings to /workspace/."""

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}


async def main():
    graph = researcher.compile()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            "Read the project brief, load the market-sizing skill, "
            "then write an initial analysis to /workspace/analysis.md"
        )]}
    )

    print(result["messages"][-1].content[:500])

    # Check what the agent wrote to the shared VFS
    analysis = vfs.read("/workspace/analysis.md")
    if analysis:
        print(f"\n--- Agent wrote to /workspace/analysis.md ---\n{analysis[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
