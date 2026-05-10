# Daytona example

[**Daytona**](https://www.daytona.io) is a cloud sandbox platform that gives
each agent run an isolated container with full filesystem and shell access.
The `daytona-sdk` Python client provisions a sandbox on demand, and
`DaytonaBackend` adapts that sandbox to `FilesystemProtocol` + `SandboxProtocol`
so the agent gets `Read`/`Write`/`Edit`/`Glob`/`Grep` plus `Bash`.

Use it when the agent needs a real Linux environment with arbitrary tooling
(`make`, `npm`, `git`, custom binaries), but you don't want to expose the host
machine. Each run gets a fresh container; teardown is one API call.

## What this example shows

- Provisioning a Daytona sandbox via the SDK (`Daytona(config).create()`).
- Wrapping the sandbox with `DaytonaBackend` and seeding the agent's
  `.agentkit/` tree via `read_tree(...)` + `backend.upload(...)` — a single
  multipart upload, not N round-trips.
- Gating `Bash` with `DEFAULT_RULESET` so shell invocations require human
  approval before they run.
- Cleaning up the sandbox in a `try/finally` to avoid leaking ephemeral
  resources.

## Prerequisites

```bash
pip install langchain-agentkit[daytona] langchain-openai
export DAYTONA_API_KEY=...
export DAYTONA_API_URL=...   # optional; defaults to Daytona Cloud
export OPENAI_API_KEY=...
```

## Run

```bash
python examples/daytona/agent.py
```

The sandbox is created on startup and deleted on exit. Iterate freely without
worrying about host-side residue.
