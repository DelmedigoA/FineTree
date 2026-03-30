#!/usr/bin/env bash
# Load all annotation data into SQLite and launch SQLPad for exploration.
# Usage: bash db/explore.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python3"

echo "==> Loading data..."
"$PYTHON" "$SCRIPT_DIR/load_db.py" --out "$SCRIPT_DIR/finetree.db"

echo ""
echo "==> Starting SQLPad at http://localhost:3000"
echo "    (Press Ctrl+C to stop)"
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up
