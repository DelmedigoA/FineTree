#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
CONFIG_PATH="${CONFIG_PATH:-configs/finetune_qwen35a3_vl.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
MULTI_GPU="${MULTI_GPU:-auto}" # auto|0|1
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

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

gpu_count="1"
if command -v nvidia-smi >/dev/null 2>&1; then
  detected_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -n "${detected_gpus}" && "${detected_gpus}" =~ ^[0-9]+$ && "${detected_gpus}" -ge 1 ]]; then
    gpu_count="${detected_gpus}"
  fi
fi

multi_gpu_enabled="0"
case "${MULTI_GPU}" in
  1|true|TRUE|yes|YES)
    multi_gpu_enabled="1"
    ;;
  0|false|FALSE|no|NO)
    multi_gpu_enabled="0"
    ;;
  auto|AUTO)
    if [[ "${gpu_count}" -gt 1 ]]; then
      multi_gpu_enabled="1"
    fi
    ;;
  *)
    echo "Invalid MULTI_GPU value: ${MULTI_GPU}. Use auto, 0, or 1."
    exit 4
    ;;
esac

if [[ "${multi_gpu_enabled}" == "1" && "${gpu_count}" -lt 2 ]]; then
  echo "MULTI_GPU requested but only ${gpu_count} GPU detected."
  exit 4
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="${gpu_count}"
fi

train_cmd=(finetree-ft-train --config "${CONFIG_PATH}")
if [[ "${multi_gpu_enabled}" == "1" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    train_cmd=(
      torchrun
      --standalone
      --nproc_per_node "${NPROC_PER_NODE}"
      --module finetree_annotator.finetune.trainer_unsloth
      --config "${CONFIG_PATH}"
    )
  else
    train_cmd=(
      python -m torch.distributed.run
      --standalone
      --nproc_per_node "${NPROC_PER_NODE}"
      --module finetree_annotator.finetune.trainer_unsloth
      --config "${CONFIG_PATH}"
    )
  fi
fi

echo "Starting fine-tune on ${gpu_count} visible GPU(s). Log: ${log_file}"
echo "Launch mode: $([[ "${multi_gpu_enabled}" == "1" ]] && echo "multi-gpu (${NPROC_PER_NODE} proc)" || echo "single-gpu")"
echo "Train command: ${train_cmd[*]}"
"${train_cmd[@]}" 2>&1 | tee "${log_file}"

echo "Training finished."
