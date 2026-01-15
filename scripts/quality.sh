#!/bin/bash
# Run all code quality checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Determine command prefix
if command -v uv &> /dev/null; then
    RUN_CMD="uv run"
else
    RUN_CMD=""
fi

echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo

echo "1. Checking code formatting with black..."
$RUN_CMD black --check backend/ main.py
echo "   Formatting check passed!"
echo

echo "2. Running tests..."
$RUN_CMD pytest backend/tests/ -v
echo "   Tests passed!"
echo

echo "=========================================="
echo "All quality checks passed!"
echo "=========================================="
