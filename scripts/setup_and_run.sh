#!/bin/bash
# Quick setup for a fresh machine (M2 Air, etc.)
# Usage: bash scripts/setup_and_run.sh [--only mangosteen_curated,mangosteen_web,mc4]

set -e

echo "=== Setting up Nirasa ==="

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -q datasets datasketch tqdm fire

echo "Setup done!"
echo ""
echo "=== Running clean + dedup ==="
PYTHONPATH="$(pwd)" .venv/bin/python scripts/clean_and_dedup_all.py "$@"
