# BubblewrapBackend integration verification image.
#
# Used by ./scripts/test-in-docker.sh from the repo root. Provides a
# Linux environment with bubblewrap installed so the integration matrix
# can actually run the bubblewrap branch (which is skipped on macOS / on
# Linux without bwrap). Not used in CI yet; meant for local pre-merge
# verification.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

# bubblewrap: the binary under test.
# build-essential + libseccomp-dev + pkg-config: pyseccomp build deps.
# git + ca-certificates: uv/pip installs from VCS or HTTPS indices.
# python3 (apt): used INSIDE the bwrap sandbox by the edit script.
# bwrap only bind-mounts /usr, /bin, /lib so it can't see uv's
# Python in /root/.local/share/uv. The sandbox python3 doesn't run
# the project test suite — that's the uv-managed 3.14 outside.
# Two Pythons coexist: apt's for the sandbox guest, uv's for the host.
# Note: we install the test runner Python via uv (matching the host's
# .python-version) rather than relying on apt's, because the project
# pins 3.14 and Python-stdlib behavior differences across versions
# (e.g. typing.get_type_hints accepting functools.partial) surface as
# spurious test failures when the runner version drifts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        build-essential \
        git \
        libseccomp-dev \
        pkg-config \
        python3 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager + version manager).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /work

# Copy pyproject + version pin first so layer caching wins on source-only changes.
COPY pyproject.toml README.md .python-version ./
COPY src ./src
COPY tests ./tests

# Sync with the bubblewrap extra so default_seccomp_program() can
# load pyseccomp. uv reads .python-version and downloads the pinned
# interpreter when the host one isn't available.
RUN uv sync --extra dev --extra bubblewrap

# Default command: full unit + integration suite. ``test-in-docker.sh``
# captures the output to tests/docker/last-run.log on the host.
CMD ["uv", "run", "pytest", "tests/unit/", "tests/integration/test_backend_protocol.py", "tests/integration/test_bubblewrap_security.py", "-v"]
