#!/usr/bin/env bash
# Build the bubblewrap test image and run the suite inside Linux.
#
# This is the canonical way to verify BubblewrapBackend behavior on a
# non-Linux dev machine. Usage:
#
#   ./scripts/test-in-docker.sh              # build + run full suite
#   ./scripts/test-in-docker.sh -- -v -k foo # forward args to pytest
#
# Exits non-zero on any failure. Captures the run output to
# tests/docker/last-run.log so the result can be reviewed after the
# container exits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="langchain-agentkit-bwrap-test:local"
LOG="$REPO_ROOT/tests/docker/last-run.log"

echo "==> Building image: $IMAGE"
docker build \
    --file "$REPO_ROOT/tests/docker/Dockerfile" \
    --tag "$IMAGE" \
    "$REPO_ROOT"

# Forward extra args to pytest if provided after a literal `--`.
PYTEST_ARGS=()
if [[ "$#" -gt 0 && "$1" == "--" ]]; then
    shift
    PYTEST_ARGS=("$@")
fi

echo "==> Running tests in container (output -> $LOG)"
mkdir -p "$(dirname "$LOG")"

# --privileged is required for bwrap to construct a fresh /proc mount
# inside its PID namespace. Docker Desktop's macOS VM masks the
# capabilities needed for procfs mounts even with seccomp/apparmor
# unconfined, and the privilege drop only applies to the *outer*
# Docker container — the inner bwrap sandbox still applies its own
# user namespace, capability drop (--cap-drop ALL), and seccomp
# filter when configured. So running the test container privileged
# does not weaken what BubblewrapBackend itself enforces.
#
# On a bare-metal Linux Docker host you can usually drop --privileged
# and run with just `--security-opt seccomp=unconfined`, but Docker
# Desktop on macOS (LinuxKit-based VM) requires the full grant.
DOCKER_RUN_FLAGS=(
    --rm
    --privileged
)

if [[ "${#PYTEST_ARGS[@]}" -gt 0 ]]; then
    docker run "${DOCKER_RUN_FLAGS[@]}" "$IMAGE" uv run pytest "${PYTEST_ARGS[@]}" 2>&1 | tee "$LOG"
else
    docker run "${DOCKER_RUN_FLAGS[@]}" "$IMAGE" 2>&1 | tee "$LOG"
fi
