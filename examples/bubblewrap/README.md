# Bubblewrap example

[**Bubblewrap**](https://github.com/containers/bubblewrap) (`bwrap`) is a
Linux-native unprivileged sandboxing tool that runs commands inside fresh
user-namespace processes with their own mount, PID, and network namespaces. No
container runtime, no daemon, no remote API — just a system binary that's
already shipping in most Linux distros. `BubblewrapBackend` invokes `bwrap`
per call to provide `FilesystemProtocol` + `SandboxProtocol` over a host
directory bind-mounted as `/workspace`.

Use it when running multiple agents on a shared Linux host: each call gets
process and resource isolation (seccomp filtering, cgroup memory/PID limits,
`fsize` rlimits) without paying for a remote sandbox.

## What this example shows

- Constructing a `BubblewrapBackend` with the default seccomp program,
  `CgroupLimits` (1 GiB memory, 64 PIDs), and `ResourceLimits` (100 MiB
  filesize cap).
- Bind-mounting a host directory as `/workspace` — the example uses a
  `TemporaryDirectory` for self-cleanup; production deployments typically
  point at a per-session directory or an AgentFS-via-FUSE mount.
- Gating `Bash` with `DEFAULT_RULESET` so shell invocations require human
  approval.
- A runtime guard for non-Linux hosts (BubblewrapBackend cannot be
  instantiated on macOS or Windows).

## Prerequisites

Linux with unprivileged user namespaces enabled, plus:

```bash
apt-get install bubblewrap                       # or your distro equivalent
pip install langchain-agentkit[bubblewrap] langchain-openai
export OPENAI_API_KEY=...
```

The `[bubblewrap]` extra ships `pyseccomp` for the default seccomp profile.

## Run (Linux only)

```bash
python examples/bubblewrap/agent.py
```

For multi-tenant production deployments, pair with cgroups (`systemd-run`) and
a per-session host directory; see `BubblewrapBackend` constructor docs for the
full hardening surface.
