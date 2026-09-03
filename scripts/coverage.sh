#!/usr/bin/env bash
# scripts/coverage.sh
# Produce the honest coverage number for the library test root.
# Relies on [tool.coverage.*] config in pyproject.toml (do NOT re-specify --cov= sources here).
# DOES NOT enforce any coverage threshold; CI coverage job enforces the floor.
set -euo pipefail

echo "Running coverage across tests/ ..."
uv run --no-sync pytest -m "not gpu" \
    --cov \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-report=html \
    -q "$@"

echo ""
echo "Coverage XML written to: coverage.xml"
echo "Coverage HTML written to: htmlcov/"
