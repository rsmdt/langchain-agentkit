# ruff: noqa: N801, N805
"""Filesystem — give agents file tools on the real OS filesystem.

FilesystemExtension provides Read, Write, Edit, Glob, Grep, LS, and
MultiEdit tools operating on the OS filesystem via OSBackend.

By default it uses the current working directory. Pass ``root=`` to
scope the agent to a specific directory (with path traversal prevention).
"""

import asyncio
import json
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import FilesystemExtension, agent

# --- Create a workspace with pre-populated data ---

workspace = Path(tempfile.mkdtemp(prefix="agentkit_"))

# 1. Write config directly
(workspace / "data").mkdir()
(workspace / "data" / "config.json").write_text(json.dumps({
    "app_name": "MyApp",
    "version": "2.1.0",
    "features": {"dark_mode": True, "notifications": False},
}, indent=2))

# 2. Copy from real filesystem
readme_path = Path(__file__).parent.parent / "README.md"
if readme_path.exists():
    (workspace / "docs").mkdir()
    (workspace / "docs" / "README.md").write_text(readme_path.read_text())

# 3. Generate structured data programmatically
(workspace / "reports").mkdir()
for quarter in range(1, 4):
    csv = f"metric,value\nrevenue,{quarter * 1000}\ngrowth,{quarter * 5}%"
    (workspace / "reports" / f"q{quarter}_2024.csv").write_text(csv)


# --- Create an agent with filesystem access ---

class analyst(agent):
    llm = ChatOpenAI(model="gpt-4o")
    extensions = [FilesystemExtension(root=workspace)]
    prompt = """\
You are a data analyst. You have access to a filesystem with
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
