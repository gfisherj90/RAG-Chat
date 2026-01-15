#!/bin/bash
# Run pytest tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Running tests..."

if command -v uv &> /dev/null; then
    uv run pytest backend/tests/ -v "$@"
else
    pytest backend/tests/ -v "$@"
fi

echo "Tests complete!"
