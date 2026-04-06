#!/usr/bin/env bash
# .devcontainer/post-create.sh
#
# Runs once after the devcontainer is created.
# Sets up the uv virtual environment and installs all workspace dependencies.

set -euo pipefail

# Always run from the workspace root, regardless of how this script is invoked.
cd "$(dirname "$0")/.."

echo "→ Creating virtual environment with uv..."
uv venv .venv --python 3.11

echo "→ Installing workspace packages..."
# Install the shared library and all service packages in editable mode.
# --all-extras pulls in optional deps (e.g. torch stubs for type checking).
uv pip install \
    --python .venv/bin/python \
    -e "shared[dev]" \
    -e "services/frame-reader[dev]" \
    -e "services/inference-worker[dev]" \
    -e "services/frame-archiver[dev]"

echo "→ Installing pre-commit hooks..."
uv run --python .venv/bin/python pre-commit install

echo ""
echo "✓ Dev environment ready."
echo "  Interpreter : $(pwd)/.venv/bin/python"
echo "  Run tests   : make test"
echo "  Lint        : make lint"
echo "  Exec into GPU service: bash scripts/run-in-service.sh frame-reader"
