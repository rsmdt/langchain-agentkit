# ruff: noqa: N805
"""Bubblewrap-backed agent — Linux-local sandboxed shell access.

``BubblewrapBackend`` runs every operation inside a per-call ``bwrap``
user-namespace process. ``root`` is any host directory bind-mounted
into the sandbox as ``/workspace``. This example mirrors the ``daytona``
example but with a local Linux sandbox instead of a remote container —
ideal for multi-tenant Linux hosts where each agent run gets process,
filesystem, and resource isolation without a cloud round-trip.

Bash requires human approval via the DEFAULT_RULESET permission preset
because the sandbox shell can still mutate ``/workspace`` content.

Prerequisites:
    - Linux host with unprivileged user namespaces enabled
    - ``apt-get install bubblewrap`` (or distro equivalent)
    - ``pip install langchain-agentkit[bubblewrap] langchain-openai``
      (the [bubblewrap] extra ships pyseccomp for the default seccomp
      program; on non-Linux installs the marker excludes it)

Environment:
    OPENAI_API_KEY — OpenAI API key

Usage (Linux only):
    python examples/bubblewrap/agent.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import (
    Agent,
    AgentsExtension,
    FilesystemExtension,
    SkillsExtension,
)
from langchain_agentkit.backends import read_tree
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
    if sys.platform != "linux":
        raise RuntimeError(
            "BubblewrapBackend is Linux-only. Use OSBackend for local development "
            "on macOS/Windows or DaytonaBackend for a cloud sandbox."
        )

    # Lazy import — fails loud on non-Linux or when bubblewrap isn't installed.
    from langchain_agentkit.backends.bubblewrap import (
        BubblewrapBackend,
        CgroupLimits,
        ResourceLimits,
        default_seccomp_program,
    )

    with tempfile.TemporaryDirectory(prefix="bwrap-agent-") as tmpdir:
        backend = BubblewrapBackend(
            root=tmpdir,
            seccomp_program=default_seccomp_program(),
            cgroup_limits=CgroupLimits(
                memory_max_bytes=1 * 1024**3,  # 1 GiB
                pids_max=64,
            ),
            rlimits=ResourceLimits(fsize_bytes=100 * 1024**2),  # 100 MiB
        )
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


if __name__ == "__main__":
    # Requires OPENAI_API_KEY in the environment; Linux host with bubblewrap.
    asyncio.run(main())
