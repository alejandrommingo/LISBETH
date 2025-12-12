#!/bin/bash
set -e

echo "=== Running Static Analysis ==="
# Check valid imports by running help
.venv/bin/python -m src.news_harvester.cli --help > /dev/null
echo "CLI Help: OK"

echo "=== Running Deterministic Tests ==="
# Run the newly created tests
if [ -x ".venv/bin/pytest" ]; then
    .venv/bin/pytest tests/test_phase1_qa.py -v
else
    echo "pytest not found in venv, attempting to run via python -m pytest"
    .venv/bin/python -m pytest tests/test_phase1_qa.py -v
fi

echo "=== QA Phase 1 Verification Completed ==="
