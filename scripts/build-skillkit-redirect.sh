#!/usr/bin/env bash
# Build the langchain-skillkit redirect package.
#
# This creates a thin PyPI package that depends on langchain-agentkit,
# so existing `pip install langchain-skillkit` users get redirected.
#
# Usage:
#   ./scripts/build-skillkit-redirect.sh          # build only
#   ./scripts/build-skillkit-redirect.sh publish   # build + publish
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$(mktemp -d)"

trap 'rm -rf "$BUILD_DIR"' EXIT

# Copy config and README
cp "$PROJECT_ROOT/pyproject-skillkit.toml" "$BUILD_DIR/pyproject.toml"
cp "$PROJECT_ROOT/README-skillkit.md" "$BUILD_DIR/README-skillkit.md"

# Create minimal __init__.py with deprecation warning
mkdir -p "$BUILD_DIR/langchain_skillkit"
cat > "$BUILD_DIR/langchain_skillkit/__init__.py" << 'PYEOF'
"""Deprecated — use langchain_agentkit instead."""
import warnings

warnings.warn(
    "langchain_skillkit has been renamed to langchain_agentkit. "
    "Update your imports: 'from langchain_agentkit import ...'",
    DeprecationWarning,
    stacklevel=2,
)
PYEOF

# Build
echo "Building langchain-skillkit redirect package..."
cd "$BUILD_DIR"
uv build

# Copy artifacts back
mkdir -p "$PROJECT_ROOT/dist"
cp "$BUILD_DIR"/dist/langchain_skillkit-* "$PROJECT_ROOT/dist/"

echo ""
echo "Built:"
ls -la "$PROJECT_ROOT"/dist/langchain_skillkit-*

if [[ "${1:-}" == "publish" ]]; then
    echo ""
    echo "Publishing to PyPI..."
    uv publish "$PROJECT_ROOT"/dist/langchain_skillkit-*
    echo "Published!"
fi
