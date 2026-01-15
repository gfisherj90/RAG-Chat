#!/bin/bash
# Check Python code formatting with black (no changes)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Checking Python code formatting..."

if command -v uv &> /dev/null; then
    uv run black --check backend/ main.py
else
    black --check backend/ main.py
fi

echo "All files are properly formatted!"
