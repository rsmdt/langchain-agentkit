"""Unit tests for BubblewrapBackend.

These tests run on any platform — they don't actually spawn ``bwrap``.
End-to-end tests against a real bwrap process live in
``tests/integration/test_backend_protocol.py`` (parameterized matrix)
and ``tests/integration/test_bubblewrap_security.py`` (security
specifics), and execute only on Linux with bubblewrap installed.

Coverage here:

- Path resolution (``_resolve``)
- Constructor validation (root + extra_ro_binds + deny-list)
- bwrap argv construction (no-net, with-net, with extra binds, with
  seccomp placeholder, with cgroup wrapping)
- ``default_seccomp_program`` (skipped when pyseccomp absent)
- Helper dataclasses (``CgroupLimits.is_empty``, ``ResourceLimits``)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from langchain_agentkit.backends.bubblewrap import (
    BubblewrapBackend,
    CgroupLimits,
    ResourceLimits,
    _shell_quote,
    _validate_bind_path,
    default_seccomp_program,
)

# Many tests inspect deny-list behavior using real host paths like /etc,
# which only resolve as expected on Linux. macOS has /etc -> /private/etc
# symlinks that defeat the prefix check after realpath. The backend is
# Linux-only in production (bwrap detection fails elsewhere), so these
# tests are skipped on non-Linux. Cross-platform deny-list logic is
# covered separately via a mocked-realpath unit test.
linux_only = pytest.mark.skipif(
    sys.platform != "linux",
    reason="deny-list test depends on Linux realpath semantics",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _workspace():
    """Yield a real temp directory usable as ``root`` (passes deny-list)."""
    with tempfile.TemporaryDirectory(prefix="bwrap-test-") as tmpdir:
        yield tmpdir


def _make_backend(
    root: str,
    *,
    cgroup_limits: CgroupLimits | None = None,
    rlimits: ResourceLimits | None = None,
    seccomp_program: bytes | None = None,
    allow_network: bool = False,
    extra_ro_binds: tuple[tuple[str, str], ...] = (),
    systemd_run_available: bool = True,
) -> BubblewrapBackend:
    """Build a backend with bwrap detection short-circuited.

    Forces ``bwrap_path`` so the constructor doesn't need a real bwrap
    on PATH; controls ``systemd-run`` discovery via ``shutil.which``
    patching so tests can exercise both branches.
    """
    real_which = shutil.which

    def _which(cmd: str) -> str | None:
        if cmd == "systemd-run":
            return "/usr/bin/systemd-run" if systemd_run_available else None
        return real_which(cmd)

    with patch("langchain_agentkit.backends.bubblewrap.shutil.which", side_effect=_which):
        return BubblewrapBackend(
            root,
            bwrap_path="/usr/bin/bwrap",
            cgroup_limits=cgroup_limits,
            rlimits=rlimits,
            seccomp_program=seccomp_program,
            allow_network=allow_network,
            extra_ro_binds=extra_ro_binds,
        )


# ---------------------------------------------------------------------------
# _shell_quote
# ---------------------------------------------------------------------------


class TestShellQuote:
    def test_plain_string(self):
        assert _shell_quote("hello") == "'hello'"

    def test_with_single_quote(self):
        # POSIX 'safe' single-quote: 'it'"'"'s'
        assert _shell_quote("it's") == "'it'\"'\"'s'"

    def test_empty(self):
        assert _shell_quote("") == "''"


# ---------------------------------------------------------------------------
# _validate_bind_path
# ---------------------------------------------------------------------------


class TestValidateBindPath:
    def test_accepts_normal_dir(self):
        with _workspace() as ws:
            assert _validate_bind_path(ws, kind="root") == os.path.realpath(ws)

    def test_rejects_root(self):
        with pytest.raises(ValueError, match="host root"):
            _validate_bind_path("/", kind="root")

    @linux_only
    @pytest.mark.parametrize(
        "denied",
        ["/proc", "/sys", "/dev", "/etc", "/home", "/root", "/run"],
    )
    def test_rejects_deny_list(self, denied):
        if not os.path.isdir(denied):
            pytest.skip(f"{denied} not present on this host")
        with pytest.raises(ValueError, match="deny-list"):
            _validate_bind_path(denied, kind="root")

    def test_rejects_under_deny_prefix_via_mocked_realpath(self, tmp_path):
        # Cross-platform deny-list coverage: stub realpath so the
        # validator sees a path under /etc regardless of the host's
        # actual /etc layout (macOS resolves /etc -> /private/etc and
        # would otherwise sidestep the check).
        target = tmp_path / "fake"
        target.mkdir()
        with (
            patch(
                "langchain_agentkit.backends.bubblewrap.os.path.realpath",
                return_value="/etc/hosts",
            ),
            pytest.raises(ValueError, match="deny-list"),
        ):
            _validate_bind_path(str(target), kind="root")

    def test_rejects_missing(self):
        with pytest.raises(FileNotFoundError):
            _validate_bind_path("/nonexistent/path/xyz", kind="root")

    def test_rejects_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            _validate_bind_path(str(f), kind="root")

    def test_resolves_symlink(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        # Symlink resolves to its target — closes CVE-2024-42472 at the
        # validation layer.
        assert _validate_bind_path(str(link), kind="root") == str(target)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_missing_bwrap_raises(self):
        with (
            patch("langchain_agentkit.backends.bubblewrap.shutil.which", return_value=None),
            _workspace() as ws,
            pytest.raises(FileNotFoundError, match="bwrap"),
        ):
            BubblewrapBackend(ws)

    def test_root_realpath_resolved(self, tmp_path):
        target = tmp_path / "real"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target)
        b = _make_backend(str(link))
        assert b.root == str(target)

    def test_extra_ro_binds_validated_via_mocked_realpath(self, tmp_path):
        # Stub realpath so we test deny-list logic deterministically
        # across platforms (see TestValidateBindPath above for rationale).
        with _workspace() as ws:
            valid_real = os.path.realpath(ws)

            def _fake_realpath(p):
                # Backend's root passes through unchanged; the bind
                # source resolves into the deny-list.
                if p in (ws, valid_real):
                    return valid_real
                return "/etc/sneaky"

            with (
                patch(
                    "langchain_agentkit.backends.bubblewrap.os.path.realpath",
                    side_effect=_fake_realpath,
                ),
                pytest.raises(ValueError, match="deny-list"),
            ):
                _make_backend(ws, extra_ro_binds=(("/var/skills", "/skills"),))

    def test_extra_ro_binds_relative_sandbox_path_rejected(self):
        with _workspace() as src, _workspace() as ws:  # noqa: SIM117
            with pytest.raises(ValueError, match="absolute"):
                _make_backend(ws, extra_ro_binds=((src, "skills"),))

    def test_extra_ro_binds_collision_with_workspace(self):
        with _workspace() as src, _workspace() as ws:  # noqa: SIM117
            with pytest.raises(ValueError, match="collides"):
                _make_backend(ws, extra_ro_binds=((src, "/workspace/skills"),))

    def test_allow_network_warns(self):
        with _workspace() as ws, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_backend(ws, allow_network=True)
            assert any("network namespace" in str(w.message) for w in caught)

    def test_cgroup_without_systemd_run_warns(self):
        with _workspace() as ws, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_backend(
                ws,
                cgroup_limits=CgroupLimits(memory_max_bytes=1024 * 1024),
                systemd_run_available=False,
            )
            assert any("systemd-run is not on PATH" in str(w.message) for w in caught)

    def test_cgroup_empty_no_warning(self):
        with _workspace() as ws, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_backend(
                ws,
                cgroup_limits=CgroupLimits(),  # all None — empty
                systemd_run_available=False,
            )
            assert not any("systemd-run" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# _resolve — virtual path → in-sandbox path
# ---------------------------------------------------------------------------


class TestResolve:
    def test_root_path(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            assert b._resolve("/") == "/workspace"
            assert b._resolve("") == "/workspace"

    def test_relative_path(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            assert b._resolve("foo.txt") == "/workspace/foo.txt"

    def test_workspace_rooted_legacy(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            assert b._resolve("/foo.txt") == "/workspace/foo.txt"

    def test_already_under_workspace(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            assert b._resolve("/workspace/foo.txt") == "/workspace/foo.txt"

    def test_traversal_rejected(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            with pytest.raises(PermissionError):
                b._resolve("../../etc/passwd")

    def test_normalized(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            assert b._resolve("/foo/./bar/../baz.txt") == "/workspace/foo/baz.txt"


# ---------------------------------------------------------------------------
# _bwrap_argv — argv construction
# ---------------------------------------------------------------------------


def _split_at(argv: list[str], sentinel: str) -> tuple[list[str], list[str]]:
    """Split argv at first sentinel occurrence — for finding ``--``."""
    idx = argv.index(sentinel)
    return argv[:idx], argv[idx + 1 :]


class TestBwrapArgv:
    def test_minimal_argv_shape(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            argv = b._bwrap_argv("echo hi", "/workspace")
            # bwrap binary is the first element with no cgroup wrapping.
            assert argv[0] == "/usr/bin/bwrap"
            assert "--die-with-parent" in argv
            assert "--new-session" in argv
            assert "--unshare-pid" in argv
            assert "--unshare-user" in argv
            assert "--unshare-net" in argv
            assert "--cap-drop" in argv and "ALL" in argv
            assert "--clearenv" in argv
            assert "--chdir" in argv
            # The command must be passed via /bin/bash -lc
            tail_args = argv[-3:]
            assert tail_args == ["/bin/bash", "-lc", "echo hi"]

    def test_workspace_bound_writable(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            argv = b._bwrap_argv("true", "/workspace")
            # Find "--bind <real_root> /workspace" pair.
            idx = argv.index("--bind")
            assert argv[idx + 1] == os.path.realpath(ws)
            assert argv[idx + 2] == "/workspace"

    def test_default_ro_binds_present(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            argv = b._bwrap_argv("true", "/workspace")
            # Each of these must appear as --ro-bind <host> <sandbox>
            for path in ("/usr", "/bin", "/lib"):
                assert path in argv

    def test_no_network_unshares_net(self):
        with _workspace() as ws:
            b = _make_backend(ws, allow_network=False)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--unshare-net" in argv
            # DNS/TLS bindings only present when network IS allowed.
            assert "/etc/resolv.conf" not in argv

    def test_allow_network_omits_unshare(self):
        with _workspace() as ws, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = _make_backend(ws, allow_network=True)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--unshare-net" not in argv
            # DNS bindings present so network calls actually work.
            assert "/etc/resolv.conf" in argv

    def test_extra_ro_binds_appended(self):
        with _workspace() as src, _workspace() as ws:
            b = _make_backend(ws, extra_ro_binds=((src, "/skills"),))
            argv = b._bwrap_argv("true", "/workspace")
            # Find --ro-bind <real_src> /skills
            ro_indexes = [i for i, a in enumerate(argv) if a == "--ro-bind"]
            found = False
            for i in ro_indexes:
                if argv[i + 1] == os.path.realpath(src) and argv[i + 2] == "/skills":
                    found = True
                    break
            assert found, f"extra_ro_bind not found in argv: {argv}"

    def test_tmpfs_size_set(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            argv = b._bwrap_argv("true", "/workspace")
            # --size N --tmpfs /tmp pair — size precedes tmpfs.
            idx = argv.index("--tmpfs")
            assert argv[idx + 1] == "/tmp"
            # The size is set immediately before --tmpfs.
            assert argv[idx - 2] == "--size"
            assert int(argv[idx - 1]) > 0

    def test_clearenv_then_setenv(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--clearenv" in argv
            # PATH/HOME/LANG/TERM set via --setenv.
            setenv_keys = [argv[i + 1] for i, a in enumerate(argv) if a == "--setenv"]
            for k in ("PATH", "HOME", "TERM", "LANG"):
                assert k in setenv_keys

    def test_seccomp_placeholder_present_when_program_set(self):
        with _workspace() as ws:
            b = _make_backend(ws, seccomp_program=b"\x00" * 8)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--seccomp" in argv
            idx = argv.index("--seccomp")
            # Placeholder is replaced at spawn time with the FD number.
            assert argv[idx + 1] == "__SECCOMP_FD__"

    def test_no_seccomp_when_program_none(self):
        with _workspace() as ws:
            b = _make_backend(ws, seccomp_program=None)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--seccomp" not in argv

    def test_materialize_seccomp_fd(self):
        with _workspace() as ws:
            b = _make_backend(ws, seccomp_program=b"\x00")
            tmpl = b._bwrap_argv("true", "/workspace")
            replaced = b._materialize_seccomp_fd(tmpl, 42)
            assert "__SECCOMP_FD__" not in replaced
            assert "42" in replaced

    def test_materialize_seccomp_fd_none_passthrough(self):
        with _workspace() as ws:
            b = _make_backend(ws)
            tmpl = b._bwrap_argv("true", "/workspace")
            assert b._materialize_seccomp_fd(tmpl, None) is tmpl


# ---------------------------------------------------------------------------
# systemd-run cgroup wrapping
# ---------------------------------------------------------------------------


class TestSystemdRunWrapping:
    def test_cgroup_wraps_with_systemd_run(self):
        with _workspace() as ws:
            limits = CgroupLimits(
                memory_max_bytes=1 * 1024**3,
                cpu_quota_percent=100,
                pids_max=64,
            )
            b = _make_backend(ws, cgroup_limits=limits, systemd_run_available=True)
            argv = b._bwrap_argv("true", "/workspace")
            assert argv[0] == "/usr/bin/systemd-run"
            assert "--user" in argv
            assert "--scope" in argv
            assert "--quiet" in argv
            assert "--property=MemoryMax=1073741824" in argv
            assert "--property=CPUQuota=100%" in argv
            assert "--property=TasksMax=64" in argv
            # bwrap follows the systemd-run -- separator
            sep_idx = argv.index("--")
            assert argv[sep_idx + 1] == "/usr/bin/bwrap"

    def test_empty_cgroup_no_wrap(self):
        with _workspace() as ws:
            b = _make_backend(ws, cgroup_limits=CgroupLimits(), systemd_run_available=True)
            argv = b._bwrap_argv("true", "/workspace")
            assert argv[0] == "/usr/bin/bwrap"

    def test_cgroup_without_systemd_no_wrap(self):
        with _workspace() as ws, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = _make_backend(
                ws,
                cgroup_limits=CgroupLimits(memory_max_bytes=1024 * 1024),
                systemd_run_available=False,
            )
            argv = b._bwrap_argv("true", "/workspace")
            assert argv[0] == "/usr/bin/bwrap"

    def test_io_bandwidth_properties(self):
        with _workspace() as ws:
            limits = CgroupLimits(io_max_rbps=1_000_000, io_max_wbps=2_000_000)
            b = _make_backend(ws, cgroup_limits=limits)
            argv = b._bwrap_argv("true", "/workspace")
            assert "--property=IOReadBandwidthMax=1000000" in argv
            assert "--property=IOWriteBandwidthMax=2000000" in argv


# ---------------------------------------------------------------------------
# CgroupLimits + ResourceLimits dataclasses
# ---------------------------------------------------------------------------


class TestCgroupLimits:
    def test_is_empty_default(self):
        assert CgroupLimits().is_empty()

    def test_is_empty_false_with_one_field(self):
        assert not CgroupLimits(memory_max_bytes=1024).is_empty()
        assert not CgroupLimits(pids_max=10).is_empty()
        assert not CgroupLimits(cpu_quota_percent=50).is_empty()


class TestResourceLimits:
    def test_construction_defaults(self):
        rl = ResourceLimits()
        assert rl.fsize_bytes is None
        assert rl.nproc is None
        assert rl.address_space_bytes is None
        assert rl.nofile is None

    def test_preexec_fn_none_when_no_rlimits(self):
        with _workspace() as ws:
            b = _make_backend(ws, rlimits=None)
            assert b._preexec_fn() is None

    def test_preexec_fn_returns_callable_when_rlimits_set(self):
        with _workspace() as ws:
            b = _make_backend(
                ws,
                rlimits=ResourceLimits(fsize_bytes=1024 * 1024),
            )
            fn = b._preexec_fn()
            assert callable(fn)


# ---------------------------------------------------------------------------
# default_seccomp_program
# ---------------------------------------------------------------------------


class TestDefaultSeccompProgram:
    def test_returns_bytes_when_pyseccomp_available(self):
        try:
            import pyseccomp  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pytest.skip("pyseccomp not installed")
        program = default_seccomp_program()
        assert isinstance(program, bytes)
        assert len(program) > 0

    def test_raises_clear_error_when_pyseccomp_missing(self):
        with (
            patch.dict("sys.modules", {"pyseccomp": None}),
            pytest.raises(ImportError, match="pyseccomp"),
        ):
            default_seccomp_program()
