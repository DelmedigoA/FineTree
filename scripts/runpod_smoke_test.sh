#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
CONFIG_PATH="${CONFIG_PATH:-configs/finetune_qwen35a3_vl.yaml}"

cd "${REPO_DIR}"

if [[ -f /opt/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/venv/bin/activate
elif [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

echo "[1/3] Bootstrap runtime stack..."
./scripts/runpod_bootstrap.sh

echo "[2/3] Validate local data + preflight..."
./scripts/runpod_validate_data.sh

echo "[3/3] Trainer dry-run..."
finetree-ft-train --config "${CONFIG_PATH}" --dry-run

echo "Smoke test passed."
