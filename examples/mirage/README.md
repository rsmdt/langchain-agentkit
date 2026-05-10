# Mirage example

[**Mirage**](https://docs.mirage.strukto.ai) ([repo](https://github.com/strukto-ai/mirage))
is a unified virtual filesystem for AI agents. It mounts external services
(S3, Google Drive, Slack, Gmail, GitHub, Linear, Notion, Postgres, MongoDB,
SSH, …) side-by-side under one filesystem tree, and exposes a tree-sitter
parsed bash subset that runs in-process — so an agent can pipe
`grep alert /slack/general/*.json | wc -l` across mounts as if they were
local. `MirageBackend` wraps a pre-constructed `mirage.Workspace` and exposes
`FilesystemProtocol` + `SandboxProtocol`.

Use it when the agent's "workspace" is a composition of services rather than a
single host or container — and when cross-service pipelines using familiar
shell vocabulary are more useful than per-service SDKs or MCP servers.

## What this example shows

- Constructing a `Workspace` with **multiple resources mounted at different
  prefixes**: `DiskResource` (read-only) at `/.agentkit` and `RAMResource`
  (writable) at `/workspace`. The agent reads its config directly through
  Mirage's filesystem abstraction — no `read_tree` upload step.
- Mount-mode declarations (`MountMode.READ` / `MountMode.WRITE`) granted per
  prefix.
- The "mount, don't copy" pattern: any local or remote resource the
  application wires in becomes a path in the agent's tree.

## Prerequisites

```bash
pip install langchain-agentkit[mirage] langchain-openai
export OPENAI_API_KEY=...
```

## Run

```bash
python examples/mirage/agent.py
```

To extend this with remote resources, swap `RAMResource()` / `DiskResource(...)`
for an `S3Resource`, `SlackResource`, `GitHubResource`, etc. — the
`MirageBackend` adapter and the agent prompt don't change.
