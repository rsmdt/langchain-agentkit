# ruff: noqa: N801, N805
"""Virtual filesystem — give agents file tools with pre-loaded data.

FilesystemMiddleware provides Read, Write, Edit, Glob, and Grep tools
operating on an in-memory VirtualFilesystem. Pre-populate the VFS with
data from any source before the agent runs.

The VFS is ephemeral — files exist only during the agent's execution.
Use it to give agents structured data to work with without touching
the real filesystem.
"""

import asyncio
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import FilesystemMiddleware, VirtualFilesystem, agent

# --- Load files into the VFS from various sources ---

vfs = VirtualFilesystem()

# 1. Write content directly
vfs.write("/data/config.json", json.dumps({
    "app_name": "MyApp",
    "version": "2.1.0",
    "features": {"dark_mode": True, "notifications": False},
}, indent=2))

# 2. Load from real filesystem
readme_path = Path(__file__).parent.parent / "README.md"
if readme_path.exists():
    vfs.write("/docs/README.md", readme_path.read_text())

# 3. Load multiple files from a directory
data_dir = Path(__file__).parent.parent / "src" / "langchain_agentkit" / "prompts"
if data_dir.exists():
    for file in data_dir.glob("*.md"):
        vfs.write(f"/prompts/{file.name}", file.read_text())

# 4. Generate structured data programmatically
for i in range(3):
    quarter = i + 1
    csv = f"metric,value\nrevenue,{quarter * 1000}\ngrowth,{quarter * 5}%"
    vfs.write(f"/reports/q{quarter}_2024.csv", csv)


# --- Create an agent with filesystem access ---

class analyst(agent):
    llm = ChatOpenAI(model="gpt-4o")
    middleware = [FilesystemMiddleware(filesystem=vfs)]
    prompt = """\
You are a data analyst. You have access to a virtual filesystem with
project files. Use Glob to discover files, Read to examine them,
and Write to save your analysis results."""

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}


async def main():
    graph = analyst.compile()

    # The agent can discover and read all pre-loaded files
    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            "List all available files, then read the config and summarize it."
        )]}
    )
    print(result["messages"][-1].content[:500])


if __name__ == "__main__":
    asyncio.run(main())
