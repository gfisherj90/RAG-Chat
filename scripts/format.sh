#!/bin/bash
# Format Python code with black

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Formatting Python code with black..."

if command -v uv &> /dev/null; then
    uv run black backend/ main.py
else
    black backend/ main.py
fi

echo "Formatting complete!"
