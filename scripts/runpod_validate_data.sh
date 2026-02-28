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

echo "Checking expected data folders..."
if [[ ! -d data/annotations ]]; then
  echo "Missing data/annotations directory."
  exit 2
fi
if [[ ! -d data/pdf_images ]]; then
  echo "Missing data/pdf_images directory."
  exit 2
fi

ann_count="$(find data/annotations -type f -name '*.json' | wc -l | tr -d ' ')"
img_count="$(find data/pdf_images -type f | wc -l | tr -d ' ')"

echo "Annotation files: ${ann_count}"
echo "Image files: ${img_count}"

if [[ "${ann_count}" == "0" ]]; then
  echo "No annotation JSON files found in data/annotations."
  exit 2
fi
if [[ "${img_count}" == "0" ]]; then
  echo "No images found in data/pdf_images."
  exit 2
fi

echo "Running preflight..."
finetree-ft-preflight --config "${CONFIG_PATH}" --probe-pages 1000

echo "Data validation passed."
