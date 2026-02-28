#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
CONFIG_PATH="${CONFIG_PATH:-configs/finetune_qwen35a3_vl.yaml}"
LOG_DIR="${LOG_DIR:-logs}"

cd "${REPO_DIR}"

if [[ -f /opt/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/venv/bin/activate
elif [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

./scripts/runpod_validate_data.sh

echo "Building train/val dataset..."
finetree-ft-build-dataset --config "${CONFIG_PATH}"

if [[ ! -f data/finetune/train.jsonl ]]; then
  echo "Missing data/finetune/train.jsonl after dataset build."
  exit 3
fi

train_rows="$(wc -l < data/finetune/train.jsonl | tr -d ' ')"
val_rows="0"
if [[ -f data/finetune/val.jsonl ]]; then
  val_rows="$(wc -l < data/finetune/val.jsonl | tr -d ' ')"
fi

echo "Train rows: ${train_rows}"
echo "Val rows: ${val_rows}"

if [[ "${train_rows}" == "0" ]]; then
  echo "Training dataset is empty. Aborting."
  exit 3
fi

mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/train_${ts}.log"

echo "Starting fine-tune. Log: ${log_file}"
finetree-ft-train --config "${CONFIG_PATH}" 2>&1 | tee "${log_file}"

echo "Training finished."
