# AgentFS example

[**AgentFS**](https://pypi.org/project/agentfs-sdk/) is a content-addressed
virtual filesystem persisted in a single SQLite/libSQL `.db` file. Your agent's
entire workspace — files, directory tree, history — lives inside one portable
database that you can ship, snapshot, or version like any other artifact. There
is no native shell surface, so the backend implements `FilesystemProtocol` only
(no `Bash` tool).

Use it when agent state must persist across runs without leaking onto the host
filesystem, when you want cheap snapshots, or when the workspace needs to move
between machines as a single file.

## What this example shows

- Wrapping a pre-opened `AgentFS` instance with `AgentFSBackend`.
- Seeding the workspace via `read_tree(...)` + `backend.upload(...)` so the
  host-side `.agentkit/` tree (system prompt, sub-agents, skills) becomes the
  agent's view at `/.agentkit/` on first boot.
- Composing `AgentsExtension`, `SkillsExtension`, and `FilesystemExtension`
  without `Bash` (AgentFS has no exec surface; pair with `OSBackend` or
  `DaytonaBackend` if you need shell execution).

## Prerequisites

```bash
pip install langchain-agentkit[agentfs] langchain-openai
export OPENAI_API_KEY=...
```

## Run

```bash
python examples/agentfs/agent.py
```

The agent persists its state to `./agent-state.db` in the current directory.
Re-running picks up where it left off.
