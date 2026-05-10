"""BubblewrapBackend — locally sandboxed backend over a bind-mounted workspace.

Every operation runs inside a per-call ``bwrap`` (bubblewrap) process.
File ops are implemented as shell commands routed through ``execute()``;
this mirrors :class:`langchain_agentkit.backends.daytona.DaytonaBackend`
except the "sandbox" is a local Linux user-namespace process rather
than a remote container.

Implements ``FilesystemProtocol`` and ``SandboxProtocol`` structurally.

Threat model
============

The backend constrains both shell execution AND file I/O. An agent that
plants ``/workspace/sneaky -> /etc/passwd`` cannot exfiltrate via
``Read('/sneaky')`` because the read also runs inside the sandbox where
``/etc`` is not bound — symlink resolution dereferences inside the same
mount namespace bash sees.

What this does NOT close: ``/usr``, ``/bin``, ``/lib``, and ``/lib64``
are bind-mounted read-only from the host so bash and the dynamic
linker work. **Anything readable under those paths is reachable to the
agent.** A planted ``/workspace/x -> /usr/share/<file>`` will return
the host file's contents. Don't place secrets under those prefixes;
treat them as part of the agent's view.

For multi-tenant deployment (multiple users' agents on one host), the
namespace isolation is necessary but not sufficient. You also need:

- A ``CgroupLimits`` config to cap memory/CPU/PIDs per call (requires
  ``systemd-run`` on the host; backend warns and degrades to rlimits
  when unavailable).
- A ``seccomp_program`` to constrain the kernel-syscall surface
  (``default_seccomp_program()`` ships a conservative allow-list when
  ``langchain-agentkit[bubblewrap]`` is installed).
- A ``ResourceLimits`` config for per-process rlimits (always-on, no
  external deps).
- ``tmpfs_size_bytes`` to cap ``/tmp`` exhaustion.

Without these, one agent can fork-bomb the host or fill its memory and
take down siblings. Defaults give correctness; multi-tenant hardening
is opt-in via constructor args.

Requirements
============

- Linux host with unprivileged user namespaces enabled.
- ``bubblewrap`` package installed (``apt-get install bubblewrap``).
- For cgroup limits: ``systemd-run`` on PATH, with a user systemd manager
  (typical on interactive logins, less so in containers).
- For ``default_seccomp_program()``: ``pip install langchain-agentkit[bubblewrap]``.

Bubblewrap is Linux-only — this backend cannot be instantiated on
macOS/Windows. Use ``OSBackend`` for local development on those
platforms and ``BubblewrapBackend`` only on Linux production hosts.

Workspace source
================

``root`` is any host directory. The backend bind-mounts it as
``/workspace`` inside the sandbox; it doesn't care what produced the
directory. Common choices:

- A regular host directory.
- A tmpfs / loopback ext4 image (per-session quota).
- An ``AgentFS`` ``.db`` mounted via ``agentfs run`` (Linux/FUSE) so the
  workspace is content-addressed with cheap snapshots — see the AgentFS
  docs. The mount is the caller's responsibility.

Usage
=====

::

    from langchain_agentkit.backends.bubblewrap import (
        BubblewrapBackend, CgroupLimits, ResourceLimits,
        default_seccomp_program,
    )

    backend = BubblewrapBackend(
        root="/var/sessions/abc123",
        seccomp_program=default_seccomp_program(),
        cgroup_limits=CgroupLimits(
            memory_max_bytes=1 * 1024**3,   # 1 GiB
            cpu_quota_percent=100,           # 1 CPU
            pids_max=64,
        ),
        rlimits=ResourceLimits(fsize_bytes=100 * 1024**2),
    )
    result = await backend.read("/main.py")
    if result.error is None:
        print(result.content)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import posixpath
import re
import resource
import shutil
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_agentkit.backends.execution import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_OUTPUT_LINES,
    BoundedCapture,
    drain_stream_into,
)
from langchain_agentkit.backends.protocol import ExecuteResponse, GrepMatch
from langchain_agentkit.backends.results import (
    PROBED_TOOLS,
    EditResult,
    FileDownloadResult,
    FileUploadResult,
    ReadBytesResult,
    ReadResult,
    SandboxEnvironment,
    WriteResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# In-sandbox writable mount point. Stable across all calls.
_SANDBOX_WORKSPACE = "/workspace"

# Default per-file size cap on read_bytes/download.
_DEFAULT_MAX_BINARY_BYTES = 50 * 1024 * 1024  # 50 MiB

# Default cap for edit() input file size — guards against OOM in the
# in-sandbox python3 edit script which reads the whole file into memory.
_DEFAULT_MAX_EDIT_BYTES = 50 * 1024 * 1024  # 50 MiB

# Default tmpfs size for /tmp inside the sandbox.
_DEFAULT_TMPFS_SIZE = 128 * 1024 * 1024  # 128 MiB

# Glob output cap — bypasses the BoundedCapture 256 KiB tail because
# silent truncation corrupts the parsed list (a real bug also present
# in the upstream DaytonaBackend.glob).
_DEFAULT_MAX_GLOB_OUTPUT = 16 * 1024 * 1024  # 16 MiB

# Host-path prefixes that must never be bind-mounted into the sandbox.
# Each exposes either kernel state (PIDs/devices), runtime sockets
# (D-Bus/systemd), system credentials, or user secrets. CVE-2024-42472
# established that bind-mount sources are realpath()'d, so symlink
# games at the source side don't bypass this list.
_DENIED_BIND_HOST_PREFIXES: tuple[str, ...] = (
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/home",
    "/root",
    "/run",
    "/var/run",
    "/var/log",
    "/var/lib",
)

# Strip bash's ``setlocale`` startup warning. Same idiom as DaytonaBackend
# — the locale data isn't bound into the minimal sandbox image, so bash
# emits the warning to stdout before any in-shell ``export`` can adjust
# LC_ALL. Left unstripped it corrupts the first line of file content
# and the environment probe.
_BASH_STARTUP_WARNING_RE = re.compile(
    r"^[^:\n]*: warning: setlocale:[^\n]*\n",
    re.MULTILINE,
)


def _shell_quote(s: str) -> str:
    """POSIX single-quote escape — same idiom DaytonaBackend uses."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


# In-sandbox environment probe. Output layout (line-oriented):
#   line 1:  uname -srm
#   line 2:  $SHELL (or /bin/sh)
#   line 3:  literal "TOOLS_BEGIN" separator
#   lines 4+: one available PROBED_TOOLS entry per line
_ENV_PROBE_SEPARATOR = "TOOLS_BEGIN"
_ENV_PROBE_SCRIPT = (
    "uname -srm\n"
    'echo "${SHELL:-/bin/sh}"\n'
    f"echo {_ENV_PROBE_SEPARATOR}\n"
    "for t in " + " ".join(PROBED_TOOLS) + "; do "
    'command -v "$t" >/dev/null 2>&1 && echo "$t"; '
    "done\n"
    "exit 0"
)


# In-sandbox edit script. Receives a base64-encoded JSON payload with
# ``{path, old, new, replace_all, max_bytes}``. Emits one JSON line on
# stdout: either ``{"replacements": N}`` or ``{"error": ..., ...}``.
# The size check happens before the read so adversarial input doesn't
# OOM the python3 process.
_EDIT_SCRIPT = """
import json, base64, sys, os
d = json.loads(base64.b64decode(sys.argv[1]))
p = d['path']
mx = d['max_bytes']
try:
    sz = os.path.getsize(p)
except FileNotFoundError:
    print(json.dumps({'error': 'file_not_found'})); sys.exit(0)
except OSError as e:
    print(json.dumps({'error': 'io_error', 'message': str(e)})); sys.exit(0)
if sz > mx:
    print(json.dumps({'error': 'io_error',
                      'message': 'File too large: ' + str(sz) + ' bytes (cap ' + str(mx) + ')'}))
    sys.exit(0)
try:
    t = open(p).read()
except IsADirectoryError:
    print(json.dumps({'error': 'is_directory'})); sys.exit(0)
except UnicodeDecodeError:
    print(json.dumps({'error': 'decode_error'})); sys.exit(0)
except OSError as e:
    print(json.dumps({'error': 'io_error', 'message': str(e)})); sys.exit(0)
n = t.count(d['old'])
if n == 0:
    print(json.dumps({'error': 'old_string_not_found'})); sys.exit(0)
if n > 1 and not d['replace_all']:
    print(json.dumps({'error': 'ambiguous_match', 'occurrences': n})); sys.exit(0)
new = t.replace(d['old'], d['new']) if d['replace_all'] else t.replace(d['old'], d['new'], 1)
try:
    open(p, 'w').write(new)
except OSError as e:
    print(json.dumps({'error': 'io_error', 'message': str(e)})); sys.exit(0)
print(json.dumps({'replacements': n if d['replace_all'] else 1}))
"""


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Per-process rlimits applied via ``preexec_fn`` before bwrap exec.

    All fields are optional and ``None`` means "don't touch this limit".
    Limits apply to bwrap and (because they're inherited) to every
    process the sandbox spawns. Attributes:

    fsize_bytes: ``RLIMIT_FSIZE`` — max single-file write size. Defends
        against ``/workspace`` exhaustion when the workspace is a host
        directory without a filesystem-level quota.
    nproc: ``RLIMIT_NPROC`` — max processes for the *real uid* on the
        host. Note: PID-namespace isolation hides processes but doesn't
        cap them — this is the cap.
    address_space_bytes: ``RLIMIT_AS`` — max virtual memory for any
        single process. Coarser than cgroup ``memory.max``.
    nofile: ``RLIMIT_NOFILE`` — max open file descriptors per process.
    """

    fsize_bytes: int | None = None
    nproc: int | None = None
    address_space_bytes: int | None = None
    nofile: int | None = None


@dataclass(frozen=True, slots=True)
class CgroupLimits:
    """Cgroup v2 limits applied by wrapping bwrap with ``systemd-run``.

    Only effective when ``systemd-run`` is on PATH and a user systemd
    manager is reachable (``loginctl enable-linger`` for the worker
    user is the typical setup). When ``systemd-run`` isn't available
    the backend warns at construction and falls back to rlimits-only.

    memory_max_bytes: ``MemoryMax=`` — hard RAM cap; OOM-killed past it.
    memory_swap_max_bytes: ``MemorySwapMax=`` — swap cap.
    cpu_quota_percent: ``CPUQuota=N%`` — 100 means one full CPU.
    pids_max: ``TasksMax=`` — process-count cap inside the scope.
    io_max_rbps / io_max_wbps: ``IOReadBandwidthMax`` / ``IOWriteBandwidthMax``
        in bytes/sec. Most useful when the workspace lives on its own
        block device.
    """

    memory_max_bytes: int | None = None
    memory_swap_max_bytes: int | None = None
    cpu_quota_percent: int | None = None
    pids_max: int | None = None
    io_max_rbps: int | None = None
    io_max_wbps: int | None = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.memory_max_bytes,
                self.memory_swap_max_bytes,
                self.cpu_quota_percent,
                self.pids_max,
                self.io_max_rbps,
                self.io_max_wbps,
            )
        )


# ---------------------------------------------------------------------------
# Bind-path validation
# ---------------------------------------------------------------------------


def _validate_bind_path(host_path: str, *, kind: str) -> str:
    """Realpath ``host_path`` and reject deny-listed prefixes.

    CVE-2024-42472 made symlink-following at bind-mount sources a known
    exfil channel. We resolve to the realpath up front so a symlink
    swap mid-flight (whether accidental or adversarial) doesn't change
    the bound target without going through this validator.

    Returns the realpath. Raises ``ValueError``, ``FileNotFoundError``,
    or ``NotADirectoryError`` on rejection.
    """
    if not host_path:
        raise ValueError(f"{kind}: empty path")
    real = os.path.realpath(host_path)
    if real == "/":
        raise ValueError(f"{kind}: refusing to bind-mount the host root '/'")
    for denied in _DENIED_BIND_HOST_PREFIXES:
        denied_prefix = denied.rstrip("/") + "/"
        if real == denied or real.startswith(denied_prefix):
            raise ValueError(
                f"{kind}: refusing to bind-mount {real!r} — matches deny-list prefix {denied!r}"
            )
    if not os.path.exists(real):
        raise FileNotFoundError(f"{kind}: path does not exist: {host_path}")
    if not os.path.isdir(real):
        raise NotADirectoryError(f"{kind}: path is not a directory: {host_path}")
    return real


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------


class BubblewrapBackend:
    """Locally sandboxed backend — every operation runs inside bubblewrap.

    Args:
        root: Host path bind-mounted as ``/workspace`` inside the
            sandbox. Realpath-validated at construction; rejected if
            it falls under a deny-list prefix.
        bwrap_path: Path to the ``bwrap`` binary. Defaults to
            ``shutil.which('bwrap')``.
        allow_network: When ``True`` the sandbox shares the host's
            network namespace — meaning every loopback service, every
            cloud metadata service (169.254.169.254), and every
            internal-VPC address is reachable to the agent. Default
            ``False`` (no network).
        extra_ro_binds: Additional read-only bind mounts as
            ``(host_path, sandbox_path)`` tuples. Each host path is
            realpath-validated against the deny-list. Sandbox paths
            must be absolute and not collide with ``/workspace``.
        env: Sandbox environment variables. Defaults to a literal
            constant set (PATH, HOME, TERM, LANG) — never reads
            ``os.environ`` so caller-process secrets cannot leak.
        seccomp_program: Optional compiled BPF program (bytes) installed
            via ``--seccomp``. Use :func:`default_seccomp_program` for a
            conservative starter profile.
        cgroup_limits: Optional cgroup v2 limits applied via
            ``systemd-run --user --scope``. Backend warns and degrades
            when ``systemd-run`` isn't available.
        rlimits: Optional per-process rlimits applied via ``preexec_fn``.
        tmpfs_size_bytes: Size cap for the in-sandbox ``/tmp`` tmpfs.
        default_timeout: Default per-call timeout in seconds.
        max_output_bytes / max_output_lines: ``BoundedCapture`` caps on
            text-output operations.
        max_binary_read_bytes: Hard cap on ``read_bytes`` and
            ``download`` per-file size.
        max_edit_file_bytes: Hard cap on ``edit`` input-file size,
            enforced inside the sandbox before the file is read into
            memory.
        max_glob_output_bytes: Cap on ``glob`` output bytes. Higher
            than ``max_output_bytes`` because glob bypasses
            BoundedCapture (silent truncation would corrupt results).

    Raises:
        FileNotFoundError: If ``bwrap`` isn't on PATH and no
            ``bwrap_path`` was supplied.
        ValueError, FileNotFoundError, NotADirectoryError: On bind
            path validation failure (root or extra_ro_binds).
    """

    _DEFAULT_ENV: dict[str, str] = {  # noqa: RUF012  # literal default; never mutated
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": _SANDBOX_WORKSPACE,
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
    }

    def __init__(
        self,
        root: str,
        *,
        bwrap_path: str | None = None,
        allow_network: bool = False,
        extra_ro_binds: Sequence[tuple[str, str]] = (),
        env: dict[str, str] | None = None,
        seccomp_program: bytes | None = None,
        cgroup_limits: CgroupLimits | None = None,
        rlimits: ResourceLimits | None = None,
        tmpfs_size_bytes: int = _DEFAULT_TMPFS_SIZE,
        default_timeout: int = 300,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = DEFAULT_MAX_OUTPUT_LINES,
        max_binary_read_bytes: int = _DEFAULT_MAX_BINARY_BYTES,
        max_edit_file_bytes: int = _DEFAULT_MAX_EDIT_BYTES,
        max_glob_output_bytes: int = _DEFAULT_MAX_GLOB_OUTPUT,
    ) -> None:
        bwrap = bwrap_path or shutil.which("bwrap")
        if bwrap is None:
            raise FileNotFoundError(
                "bwrap not found on PATH. Install bubblewrap "
                "(`apt-get install bubblewrap`) or pass bwrap_path="
            )

        self._root = _validate_bind_path(root, kind="root")

        validated_binds: list[tuple[str, str]] = []
        for host, sandbox in extra_ro_binds:
            real_host = _validate_bind_path(host, kind=f"extra_ro_bind ({host!r})")
            if not sandbox.startswith("/"):
                raise ValueError(f"extra_ro_bind sandbox path must be absolute, got {sandbox!r}")
            sandbox_norm = posixpath.normpath(sandbox)
            if sandbox_norm == _SANDBOX_WORKSPACE or sandbox_norm.startswith(
                _SANDBOX_WORKSPACE + "/"
            ):
                raise ValueError(
                    f"extra_ro_bind sandbox path {sandbox!r} collides with "
                    f"the workspace mount at {_SANDBOX_WORKSPACE!r}"
                )
            validated_binds.append((real_host, sandbox_norm))

        if allow_network:
            warnings.warn(
                "BubblewrapBackend(allow_network=True) shares the host "
                "network namespace — the sandbox can reach loopback "
                "services, cloud metadata (169.254.169.254), and "
                "internal-VPC addresses. Disable IMDSv1 and apply "
                "host-level egress filtering before enabling in cloud.",
                stacklevel=2,
            )

        self._bwrap = bwrap
        self._allow_network = allow_network
        self._extra_ro_binds: tuple[tuple[str, str], ...] = tuple(validated_binds)
        self._env = self._DEFAULT_ENV.copy() if env is None else dict(env)
        self._seccomp_program = seccomp_program
        self._cgroup_limits = cgroup_limits
        self._rlimits = rlimits
        self._tmpfs_size_bytes = tmpfs_size_bytes
        self._default_timeout = default_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_binary_read_bytes = max_binary_read_bytes
        self._max_edit_file_bytes = max_edit_file_bytes
        self._max_glob_output_bytes = max_glob_output_bytes
        self._workdir = _SANDBOX_WORKSPACE
        self._env_cache: SandboxEnvironment | None = None

        # Detect systemd-run once. When cgroup_limits are configured
        # without systemd-run available we degrade to rlimits-only and
        # warn — failing hard would be too aggressive given that
        # rlimits still provide partial protection.
        self._systemd_run: str | None = shutil.which("systemd-run")
        if cgroup_limits is not None and not cgroup_limits.is_empty() and self._systemd_run is None:
            warnings.warn(
                "BubblewrapBackend: cgroup_limits configured but "
                "systemd-run is not on PATH — falling back to "
                "rlimits-only. Multi-tenant resource isolation will "
                "be partial. Install systemd or supply external "
                "cgroup wrapping for full enforcement.",
                stacklevel=2,
            )

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def root(self) -> str:
        """Host path bound as the workspace inside the sandbox."""
        return self._root

    # ----- Path resolution ------------------------------------------------

    def _resolve(self, path: str) -> str:
        """Resolve an LLM-supplied path to a sandbox-absolute path.

        Pure string normalization — does NOT touch the host filesystem.
        Symlinks within ``/workspace`` are resolved by the sandbox at
        I/O time, where they cannot escape the bind mount.

        Paths already under ``/workspace`` pass through. Other absolute
        paths and relative paths are interpreted as workspace-rooted
        (legacy ``/foo`` -> ``/workspace/foo``). Anything that
        normalizes outside ``/workspace`` raises ``PermissionError``.
        """
        if not path or path == "/":
            return self._workdir
        prefix = self._workdir.rstrip("/") + "/"
        if path.startswith("/"):
            normalized = posixpath.normpath(path)
            if normalized == self._workdir or normalized.startswith(prefix):
                return normalized
        cleaned = path.lstrip("/")
        candidate = posixpath.normpath(f"{self._workdir}/{cleaned}")
        if candidate != self._workdir and not candidate.startswith(prefix):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    # ----- argv construction ----------------------------------------------

    def _bwrap_argv(self, command: str, sandbox_workdir: str) -> list[str]:
        """Build the bwrap argv for a single command invocation.

        When ``cgroup_limits`` are configured and ``systemd-run`` is
        available, the bwrap call is wrapped so the entire command
        runs inside a transient cgroup v2 scope.
        """
        argv: list[str] = []

        if (
            self._cgroup_limits is not None
            and not self._cgroup_limits.is_empty()
            and self._systemd_run is not None
        ):
            argv += self._systemd_run_prefix()

        argv += [
            self._bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--unshare-user",
        ]
        if not self._allow_network:
            argv.append("--unshare-net")

        # Minimum host bindings for bash + coreutils + python3.
        argv += [
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind-try",
            "/etc/alternatives",
            "/etc/alternatives",
        ]
        if self._allow_network:
            # DNS + TLS — only useful with network on.
            argv += [
                "--ro-bind-try",
                "/etc/resolv.conf",
                "/etc/resolv.conf",
                "--ro-bind-try",
                "/etc/ssl",
                "/etc/ssl",
                "--ro-bind-try",
                "/etc/ca-certificates",
                "/etc/ca-certificates",
            ]

        argv += ["--bind", self._root, _SANDBOX_WORKSPACE]
        for host, sandbox in self._extra_ro_binds:
            argv += ["--ro-bind", host, sandbox]

        argv += [
            "--size",
            str(self._tmpfs_size_bytes),
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--uid",
            "1000",
            "--gid",
            "1000",
            "--cap-drop",
            "ALL",
            "--chdir",
            sandbox_workdir,
            "--clearenv",
        ]
        for k, v in self._env.items():
            argv += ["--setenv", k, v]
        # ``--seccomp`` consumes the FD that gets passed via pass_fds.
        # The placeholder is replaced by the actual FD number in _shell
        # because the FD is opened per-call.
        if self._seccomp_program is not None:
            argv += ["--seccomp", "__SECCOMP_FD__"]

        argv += ["--", "/bin/bash", "-lc", command]
        return argv

    def _systemd_run_prefix(self) -> list[str]:
        """Build the ``systemd-run --user --scope`` wrapper argv."""
        assert self._systemd_run is not None
        assert self._cgroup_limits is not None
        argv = [self._systemd_run, "--user", "--scope", "--quiet"]
        cl = self._cgroup_limits
        if cl.memory_max_bytes is not None:
            argv.append(f"--property=MemoryMax={cl.memory_max_bytes}")
        if cl.memory_swap_max_bytes is not None:
            argv.append(f"--property=MemorySwapMax={cl.memory_swap_max_bytes}")
        if cl.cpu_quota_percent is not None:
            argv.append(f"--property=CPUQuota={cl.cpu_quota_percent}%")
        if cl.pids_max is not None:
            argv.append(f"--property=TasksMax={cl.pids_max}")
        if cl.io_max_rbps is not None:
            argv.append(f"--property=IOReadBandwidthMax={cl.io_max_rbps}")
        if cl.io_max_wbps is not None:
            argv.append(f"--property=IOWriteBandwidthMax={cl.io_max_wbps}")
        argv.append("--")
        return argv

    # ----- seccomp + rlimits ----------------------------------------------

    def _make_seccomp_fd(self) -> int | None:
        """Open a fresh memfd containing the seccomp BPF program.

        A new FD per spawn avoids file-offset sharing between
        concurrent calls (passed FDs share the underlying open file
        with the child; if we reused one, the second call would read
        from EOF).
        """
        if self._seccomp_program is None:
            return None
        # memfd_create is Linux-only; raises AttributeError elsewhere.
        # That's fine — bwrap is Linux-only, and we never reach here
        # except after a successful bwrap spawn.
        fd: int = os.memfd_create("bwrap-seccomp", flags=0)  # type: ignore[attr-defined]
        try:
            os.write(fd, self._seccomp_program)
            os.lseek(fd, 0, os.SEEK_SET)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        return fd

    def _preexec_fn(self) -> Callable[[], None] | None:
        """Build a preexec_fn that applies rlimits in the child.

        Runs after fork() and before execve(); it's safe to call
        ``resource.setrlimit`` here because the child is single-threaded.
        Returns ``None`` when no rlimits are configured so subprocess
        skips the dangerous-with-threads code path entirely.
        """
        if self._rlimits is None:
            return None
        rl = self._rlimits

        def _apply() -> None:
            if rl.fsize_bytes is not None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (rl.fsize_bytes, rl.fsize_bytes))
            if rl.nproc is not None:
                resource.setrlimit(resource.RLIMIT_NPROC, (rl.nproc, rl.nproc))
            if rl.address_space_bytes is not None:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (rl.address_space_bytes, rl.address_space_bytes),
                )
            if rl.nofile is not None:
                resource.setrlimit(resource.RLIMIT_NOFILE, (rl.nofile, rl.nofile))

        return _apply

    # ----- internal: bounded text shell -----------------------------------

    async def _shell(
        self,
        command: str,
        *,
        stdin: bytes | None = None,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Run ``command`` in a fresh bwrap sandbox with bounded text capture.

        Used by ``execute()`` and every file-op method whose stdout is
        bounded text intended for the LLM or shell-parsing.
        """
        sandbox_workdir = self._resolve(workdir) if workdir else self._workdir
        argv_template = self._bwrap_argv(command, sandbox_workdir)

        seccomp_fd = self._make_seccomp_fd()
        argv = self._materialize_seccomp_fd(argv_template, seccomp_fd)
        pass_fds: tuple[int, ...] = (seccomp_fd,) if seccomp_fd is not None else ()

        capture = BoundedCapture(
            stdout_max_bytes=self._max_output_bytes,
            stdout_max_lines=self._max_output_lines,
            stderr_max_bytes=self._max_output_bytes,
            stderr_max_lines=self._max_output_lines,
        )
        proc: asyncio.subprocess.Process | None = None
        stdin_task: asyncio.Task[None] | None = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=(
                        asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL
                    ),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    pass_fds=pass_fds,
                    preexec_fn=self._preexec_fn(),
                )
            except OSError as exc:
                raise RuntimeError(
                    "bwrap failed to spawn (kernel may not support "
                    f"unprivileged user namespaces): {exc}"
                ) from exc

            if stdin is not None and proc.stdin is not None:
                stdin_task = asyncio.create_task(_drain_stdin(proc.stdin, stdin))

            pump = asyncio.gather(
                drain_stream_into(proc.stdout, capture.feed_stdout),
                drain_stream_into(proc.stderr, capture.feed_stderr),
                proc.wait(),
            )
            try:
                await asyncio.wait_for(pump, timeout=timeout or self._default_timeout)
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                    await asyncio.wait_for(pump, timeout=2.0)
                stdout_res, stderr_res, spill = capture.finalize()
                return ExecuteResponse(
                    output=_strip_locale(stdout_res.tail.decode("utf-8", errors="replace")),
                    stderr=stderr_res.tail.decode("utf-8", errors="replace"),
                    exit_code=-1,
                    truncated=True,
                    output_path=str(spill) if spill else None,
                    lines_dropped=stdout_res.lines_dropped + stderr_res.lines_dropped,
                    bytes_dropped=stdout_res.bytes_dropped + stderr_res.bytes_dropped,
                )
        except BaseException:
            capture.abandon()
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    proc.kill()
            raise
        finally:
            if stdin_task is not None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    await stdin_task
            if seccomp_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(seccomp_fd)

        stdout_res, stderr_res, spill = capture.finalize()
        return ExecuteResponse(
            output=_strip_locale(stdout_res.tail.decode("utf-8", errors="replace")),
            stderr=stderr_res.tail.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            truncated=spill is not None,
            output_path=str(spill) if spill else None,
            lines_dropped=stdout_res.lines_dropped + stderr_res.lines_dropped,
            bytes_dropped=stdout_res.bytes_dropped + stderr_res.bytes_dropped,
        )

    # ----- internal: raw-binary shell -------------------------------------

    async def _shell_raw(
        self,
        command: str,
        *,
        max_bytes: int,
        timeout: int | None = None,
    ) -> tuple[int, bytes, str]:
        """Run command in sandbox, capturing raw binary stdout up to ``max_bytes``.

        Returns ``(exit_code, stdout_bytes, stderr_text)``. Stderr is
        drained concurrently with stdout (NOT after) — sequential drain
        deadlocks once stderr fills its pipe buffer (~16-64 KiB).
        """
        argv_template = self._bwrap_argv(command, self._workdir)
        seccomp_fd = self._make_seccomp_fd()
        argv = self._materialize_seccomp_fd(argv_template, seccomp_fd)
        pass_fds: tuple[int, ...] = (seccomp_fd,) if seccomp_fd is not None else ()

        proc: asyncio.subprocess.Process | None = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    pass_fds=pass_fds,
                    preexec_fn=self._preexec_fn(),
                )
            except OSError as exc:
                raise RuntimeError(f"bwrap failed to spawn: {exc}") from exc

            assert proc.stdout is not None and proc.stderr is not None

            try:
                (stdout_bytes, overflowed), stderr_bytes, _ = await asyncio.wait_for(
                    asyncio.gather(
                        _read_capped(proc.stdout, max_bytes),
                        proc.stderr.read(),
                        proc.wait(),
                    ),
                    timeout=timeout or self._default_timeout,
                )
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(Exception):  # noqa: BLE001
                    await proc.wait()
                return -1, b"", "timeout"

            if overflowed:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                return -1, stdout_bytes, "size_limit_exceeded"

            return (
                proc.returncode if proc.returncode is not None else -1,
                stdout_bytes,
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except BaseException:
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(Exception):  # noqa: BLE001
                    proc.kill()
            raise
        finally:
            if seccomp_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(seccomp_fd)

    @staticmethod
    def _materialize_seccomp_fd(argv_template: list[str], fd: int | None) -> list[str]:
        """Replace the ``__SECCOMP_FD__`` placeholder with the actual FD number.

        The argv is built once per call by ``_bwrap_argv`` without
        knowing the FD number; we substitute here so unit tests can
        inspect the template independently of FD allocation.
        """
        if fd is None:
            return argv_template
        return [str(fd) if a == "__SECCOMP_FD__" else a for a in argv_template]

    # ----- SandboxProtocol ------------------------------------------------

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        workdir: str | None = None,
    ) -> ExecuteResponse:
        """Execute LLM-authored shell command in a fresh sandbox."""
        return await self._shell(command, timeout=timeout, workdir=workdir)

    async def environment(self) -> SandboxEnvironment:
        """Probe the sandbox once for OS/shell/tool inventory.

        Cached for the backend's lifetime. Probes inside the sandbox
        so the ``<env>`` block surfaced to the LLM reflects what bash
        actually sees, not the worker host's installed tools.
        """
        if self._env_cache is not None:
            return self._env_cache

        result = await self._shell(_ENV_PROBE_SCRIPT)
        os_line = "unknown"
        shell = "/bin/sh"
        tools: set[str] = set()
        if result.get("exit_code") == 0:
            lines = result.get("output", "").splitlines()
            try:
                sep_idx = lines.index(_ENV_PROBE_SEPARATOR)
            except ValueError:
                sep_idx = -1
            if sep_idx >= 2:
                os_line = lines[0].strip() or "unknown"
                shell = lines[1].strip() or "/bin/sh"
                tools = {ln.strip() for ln in lines[sep_idx + 1 :] if ln.strip()}

        self._env_cache = SandboxEnvironment(
            os=os_line,
            shell=shell,
            cwd=self._workdir,
            available_tools=frozenset(tools),
        )
        return self._env_cache

    # ----- FilesystemProtocol ---------------------------------------------

    async def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            sb_path = self._resolve(path)
        except PermissionError as exc:
            return ReadResult(error="permission_denied", error_message=str(exc))
        q = _shell_quote(sb_path)
        # Single-shell probe + read in one spawn — saves a bwrap round trip
        # vs the two-step Daytona pattern (where each step pays a network
        # RTT, so consolidation isn't worth the parsing complexity).
        check = await self._shell(
            f"if [ -d {q} ]; then echo D; elif [ -f {q} ]; then echo F; else echo M; fi"
        )
        kind = check["output"].strip()
        if kind == "M":
            return ReadResult(error="file_not_found", error_message=f"File not found: {path}")
        if kind == "D":
            return ReadResult(error="is_directory", error_message=f"Path is a directory: {path}")
        start, end = offset + 1, offset + limit
        result = await self._shell(f"sed -n '{start},{end}p' {q}")
        if result["exit_code"] != 0:
            return ReadResult(
                error="io_error",
                error_message=f"Read failed: {result.get('stderr') or result.get('output')}",
            )
        return ReadResult(content=result["output"])

    async def read_bytes(self, path: str) -> ReadBytesResult:
        try:
            sb_path = self._resolve(path)
        except PermissionError as exc:
            return ReadBytesResult(error="permission_denied", error_message=str(exc))
        q = _shell_quote(sb_path)
        check = await self._shell(
            f"if [ -d {q} ]; then echo D; "
            f"elif [ -f {q} ]; then echo F; stat -c %s {q}; "
            f"else echo M; fi"
        )
        out = check["output"].splitlines()
        kind = out[0].strip() if out else "M"
        if kind == "M":
            return ReadBytesResult(error="file_not_found", error_message=f"File not found: {path}")
        if kind == "D":
            return ReadBytesResult(
                error="is_directory", error_message=f"Path is a directory: {path}"
            )
        try:
            size = int(out[1].strip())
        except (IndexError, ValueError):
            return ReadBytesResult(error="io_error", error_message=f"Could not stat: {path}")
        if size > self._max_binary_read_bytes:
            return ReadBytesResult(
                error="io_error",
                error_message=(f"File too large: {size} bytes (cap {self._max_binary_read_bytes})"),
            )

        exit_code, content, stderr = await self._shell_raw(
            f"cat {q}", max_bytes=self._max_binary_read_bytes
        )
        if exit_code != 0:
            return ReadBytesResult(
                error="io_error",
                error_message=f"Read failed: {stderr or 'unknown error'}",
            )
        return ReadBytesResult(content=content)

    async def write(self, path: str, content: str | bytes) -> WriteResult:
        try:
            sb_path = self._resolve(path)
        except PermissionError as exc:
            return WriteResult(error="permission_denied", error_message=str(exc))
        raw = content.encode("utf-8") if isinstance(content, str) else content
        q = _shell_quote(sb_path)
        # Pipe bytes via stdin — avoids ARG_MAX entirely.
        result = await self._shell(
            f'mkdir -p "$(dirname {q})" && cat > {q}',
            stdin=raw,
        )
        if result["exit_code"] != 0:
            return WriteResult(
                error="io_error",
                error_message=f"Write failed: {result.get('stderr') or result.get('output')}",
            )
        return WriteResult(path=path, bytes_written=len(raw))

    async def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            sb_path = self._resolve(path)
        except PermissionError as exc:
            return EditResult(error="permission_denied", error_message=str(exc))
        payload = json.dumps(
            {
                "path": sb_path,
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
                "max_bytes": self._max_edit_file_bytes,
            }
        )
        encoded = base64.b64encode(payload.encode()).decode("ascii")
        result = await self._shell(
            f"python3 -c {_shell_quote(_EDIT_SCRIPT)} {_shell_quote(encoded)}"
        )
        if result["exit_code"] != 0:
            return EditResult(
                error="io_error",
                error_message=f"Edit failed: {result.get('stderr') or result.get('output')}",
            )
        try:
            data = json.loads(result["output"].strip())
        except json.JSONDecodeError:
            return EditResult(
                error="io_error",
                error_message=f"Edit produced invalid output: {result['output']!r}",
            )
        if "error" in data:
            err = data["error"]
            if err == "file_not_found":
                return EditResult(error="file_not_found", error_message=f"File not found: {path}")
            if err == "is_directory":
                return EditResult(
                    error="is_directory", error_message=f"Path is a directory: {path}"
                )
            if err == "decode_error":
                return EditResult(
                    error="decode_error",
                    error_message=f"File is not valid UTF-8: {path}",
                )
            if err == "old_string_not_found":
                return EditResult(
                    path=path,
                    error="old_string_not_found",
                    error_message=f"old_string not found in {path}",
                )
            if err == "ambiguous_match":
                occ = data.get("occurrences")
                return EditResult(
                    path=path,
                    occurrences=occ,
                    error="ambiguous_match",
                    error_message=(
                        f"old_string appears {occ} times in {path}. "
                        "Pass replace_all=True or extend old_string for uniqueness."
                    ),
                )
            return EditResult(
                error="io_error",
                error_message=data.get("message") or f"Edit failed: {err}",
            )
        return EditResult(path=path, replacements=data["replacements"])

    async def glob(self, pattern: str, path: str = "/") -> list[str]:
        """Find files matching a glob pattern.

        Bypasses ``BoundedCapture`` via ``_shell_raw`` so a large repo
        doesn't get its result list silently truncated past 256 KiB.
        Cap is configurable via ``max_glob_output_bytes`` (16 MiB
        default — large enough for tens of thousands of paths).
        """
        try:
            sb_path = self._resolve(path)
        except PermissionError:
            return []
        q = _shell_quote(sb_path)
        if "**" in pattern or "/" in pattern:
            find_pattern = pattern.replace("**", "*")
            full_pattern = f"{sb_path}/{find_pattern}"
            cmd = f"find {q} -path {_shell_quote(full_pattern)} -type f 2>/dev/null | sort"
        else:
            cmd = f"find {q} -name {_shell_quote(pattern)} -type f 2>/dev/null | sort"
        exit_code, content_bytes, _ = await self._shell_raw(
            cmd, max_bytes=self._max_glob_output_bytes
        )
        if exit_code != 0:
            return []
        text = content_bytes.decode("utf-8", errors="replace")
        if not text.strip():
            return []
        matches: list[str] = []
        for line in text.strip().splitlines():
            rel = line.removeprefix(self._workdir)
            if not rel.startswith("/"):
                rel = "/" + rel
            matches.append(rel)
        return matches

    async def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
    ) -> list[GrepMatch]:
        try:
            sb_path = self._resolve(path or "/")
        except PermissionError:
            return []
        flags = "-rni" if ignore_case else "-rn"
        cmd = f"grep {flags} {_shell_quote(pattern)} {_shell_quote(sb_path)}"
        if glob:
            cmd += f" --include={_shell_quote(glob)}"
        cmd += " 2>/dev/null || true"
        result = await self._shell(cmd)
        if not result["output"].strip():
            return []
        matches: list[GrepMatch] = []
        for line in result["output"].strip().splitlines():
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path = parts[0].removeprefix(self._workdir)
            if not file_path.startswith("/"):
                file_path = "/" + file_path
            try:
                line_num = int(parts[1])
            except ValueError:
                continue
            matches.append(GrepMatch(path=file_path, line=line_num, text=parts[2]))
        return matches

    # ----- FilesystemProtocol: bulk transfer (host-side) -----------------

    async def upload(self, files: list[tuple[str, bytes]]) -> list[FileUploadResult]:
        """Bulk upload — sequential per-file writes.

        Each file is one sandboxed write (bwrap spawn). For
        framework-driven seeding the file count is bounded; for
        thousands of files prefer a single ``tar -xf -`` pipe (not
        currently implemented).
        """
        results: list[FileUploadResult] = []
        for path, content in files:
            res = await self.write(path, content)
            if res.error is not None:
                results.append(
                    FileUploadResult(path=path, error=res.error, error_message=res.error_message)
                )
            else:
                results.append(FileUploadResult(path=path, bytes_written=res.bytes_written))
        return results

    async def download(self, paths: list[str]) -> list[FileDownloadResult]:
        """Bulk download — sequential per-file reads."""
        results: list[FileDownloadResult] = []
        for path in paths:
            res = await self.read_bytes(path)
            if res.error is not None:
                results.append(
                    FileDownloadResult(path=path, error=res.error, error_message=res.error_message)
                )
            else:
                results.append(FileDownloadResult(path=path, content=res.content))
        return results


# ---------------------------------------------------------------------------
# Default seccomp helper
# ---------------------------------------------------------------------------


# Conservative allow-list: covers bash + coreutils + python3 +
# typical agent tooling. Errs on the side of working over tight —
# operators with stronger threat models should supply their own
# program. Syscalls outside this list are killed by the kernel.
_DEFAULT_SECCOMP_ALLOW: tuple[str, ...] = (
    # File I/O
    "read",
    "write",
    "pread64",
    "pwrite64",
    "readv",
    "writev",
    "preadv",
    "pwritev",
    "preadv2",
    "pwritev2",
    "open",
    "openat",
    "openat2",
    "close",
    "close_range",
    "creat",
    "lseek",
    "_llseek",
    # Stat family
    "stat",
    "fstat",
    "lstat",
    "newfstatat",
    "statx",
    "fstatfs",
    "statfs",
    # Mmap / memory
    "mmap",
    "mmap2",
    "munmap",
    "mprotect",
    "brk",
    "mremap",
    "msync",
    "madvise",
    "mlock",
    "munlock",
    "mlockall",
    "munlockall",
    # Signal handling
    "rt_sigaction",
    "rt_sigprocmask",
    "rt_sigreturn",
    "rt_sigsuspend",
    "rt_sigpending",
    "rt_sigtimedwait",
    "rt_sigqueueinfo",
    "sigaltstack",
    # Process lifecycle
    "fork",
    "vfork",
    "clone",
    "clone3",
    "execve",
    "execveat",
    "exit",
    "exit_group",
    "wait4",
    "waitid",
    "kill",
    "tgkill",
    "tkill",
    # Process identity
    "getpid",
    "getppid",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "gettid",
    "setpgid",
    "getpgid",
    "getpgrp",
    "getsid",
    "setsid",
    "getgroups",
    "setgroups",
    "getrandom",
    # FDs and pipes
    "pipe",
    "pipe2",
    "dup",
    "dup2",
    "dup3",
    # Access checks
    "access",
    "faccessat",
    "faccessat2",
    # Directory ops
    "chdir",
    "fchdir",
    "getcwd",
    "mkdir",
    "mkdirat",
    "rmdir",
    "unlink",
    "unlinkat",
    "rename",
    "renameat",
    "renameat2",
    # Permissions
    "chmod",
    "fchmod",
    "fchmodat",
    "chown",
    "fchown",
    "fchownat",
    "lchown",
    "umask",
    # Symlinks (within the sandbox view)
    "readlink",
    "readlinkat",
    "symlink",
    "symlinkat",
    "link",
    "linkat",
    # Times
    "utime",
    "utimes",
    "utimensat",
    "futimesat",
    # Truncate
    "truncate",
    "ftruncate",
    # ioctl / fcntl
    "ioctl",
    "fcntl",
    # Polling / select
    "poll",
    "ppoll",
    "select",
    "pselect6",
    "epoll_create",
    "epoll_create1",
    "epoll_ctl",
    "epoll_wait",
    "epoll_pwait",
    "epoll_pwait2",
    # Sleeping / scheduling
    "nanosleep",
    "clock_nanosleep",
    "sched_yield",
    "sched_getaffinity",
    "sched_setaffinity",
    "sched_getparam",
    "sched_getscheduler",
    # Synchronization
    "futex",
    "set_robust_list",
    "get_robust_list",
    "set_tid_address",
    # Directory enumeration
    "getdents",
    "getdents64",
    # Misc process state
    "prctl",
    "rseq",
    "arch_prctl",
    # Time
    "sysinfo",
    "gettimeofday",
    "clock_gettime",
    "clock_getres",
    "time",
    # System info
    "uname",
    # Resource limits
    "prlimit64",
    "getrlimit",
    "setrlimit",
    "getrusage",
    # Capabilities (read-only ones; setting is blocked by --cap-drop ALL)
    "capget",
    # User namespace setup support
    "setresuid",
    "setresgid",
    "setreuid",
    "setregid",
)


def default_seccomp_program() -> bytes:
    """Build a conservative default seccomp BPF program.

    Returns a compiled BPF program (bytes) suitable for passing to
    ``BubblewrapBackend(seccomp_program=...)``. Allows ~100 syscalls
    needed for bash + coreutils + python3 + typical agent tooling;
    everything else is killed by the kernel.

    Requires the optional ``pyseccomp`` dependency:
    ``pip install langchain-agentkit[bubblewrap]`` (or ``pyseccomp`` directly).

    This is a *starting point*, not a tight defense. Operators with
    stronger threat models should write their own filter scoped to
    the specific syscalls their agent workloads need.

    Raises:
        ImportError: If ``pyseccomp`` isn't installed.
    """
    try:
        import pyseccomp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "default_seccomp_program() requires pyseccomp. "
            "Install with: pip install langchain-agentkit[bubblewrap]"
        ) from exc

    flt = pyseccomp.SyscallFilter(defaction=pyseccomp.KILL_PROCESS)
    for name in _DEFAULT_SECCOMP_ALLOW:
        with contextlib.suppress(RuntimeError, ValueError):
            # Some syscalls don't exist on every architecture; pyseccomp
            # raises on add_rule for those. Skip rather than fail —
            # the missing syscall isn't a security gap (it's not callable).
            flt.add_rule(pyseccomp.ALLOW, name)
    # ``export_bpf`` writes via fileno(), so an in-memory BytesIO won't
    # work. Use an unnamed tempfile, export, rewind, slurp.
    import tempfile

    with tempfile.TemporaryFile() as tmp:
        flt.export_bpf(tmp)
        tmp.seek(0)
        program: bytes = tmp.read()
    return program


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


async def _drain_stdin(stdin_writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write ``data`` to stdin and close it. Tolerant of pipe closure."""
    try:
        stdin_writer.write(data)
        await stdin_writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):  # noqa: BLE001
            stdin_writer.close()


async def _read_capped(stream: asyncio.StreamReader, max_bytes: int) -> tuple[bytes, bool]:
    """Drain ``stream`` up to ``max_bytes``; return (bytes, overflowed)."""
    buf = bytearray()
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return bytes(buf), False
        if len(buf) + len(chunk) > max_bytes:
            buf.extend(chunk[: max_bytes - len(buf)])
            return bytes(buf), True
        buf.extend(chunk)


def _strip_locale(text: str) -> str:
    """Strip bash setlocale warnings from output."""
    return _BASH_STARTUP_WARNING_RE.sub("", text)
