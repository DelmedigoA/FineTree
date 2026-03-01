#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
CONFIG_PATH="${CONFIG_PATH:-configs/finetune_qwen35a3_vl.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
MULTI_GPU="${MULTI_GPU:-auto}" # auto|0|1
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
PREFETCH_MODEL="${PREFETCH_MODEL:-auto}" # auto|0|1
PREFETCH_ADAPTER="${PREFETCH_ADAPTER:-auto}" # auto|0|1
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME HF_HUB_DISABLE_XET HF_HUB_ENABLE_HF_TRANSFER

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
torchrun_log_dir="${LOG_DIR}/torchrun_${ts}"
mkdir -p "${HF_HOME}"
export FINETREE_TRAIN_RUN_ID="${ts}"
export FINETREE_TRAIN_LOG_FILE="${log_file}"
export FINETREE_TORCHRUN_LOG_DIR="${torchrun_log_dir}"

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

prefetch_enabled="0"
case "${PREFETCH_MODEL}" in
  1|true|TRUE|yes|YES)
    prefetch_enabled="1"
    ;;
  0|false|FALSE|no|NO)
    prefetch_enabled="0"
    ;;
  auto|AUTO)
    if [[ "${multi_gpu_enabled}" == "1" ]]; then
      prefetch_enabled="1"
    fi
    ;;
  *)
    echo "Invalid PREFETCH_MODEL value: ${PREFETCH_MODEL}. Use auto, 0, or 1."
    exit 4
    ;;
esac

cfg_refs="$(
  python - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
model_ref = str((raw.get("model") or {}).get("base_model", "")).strip()
adapter_ref = str((raw.get("inference") or {}).get("adapter_path", "")).strip()
print(model_ref)
print(adapter_ref)
PY
)"
model_ref="$(printf '%s\n' "${cfg_refs}" | sed -n '1p')"
adapter_ref="$(printf '%s\n' "${cfg_refs}" | sed -n '2p')"

if [[ -z "${model_ref}" ]]; then
  echo "Could not resolve model.base_model from ${CONFIG_PATH}."
  exit 4
fi

prefetch_adapter_enabled="0"
case "${PREFETCH_ADAPTER}" in
  1|true|TRUE|yes|YES)
    prefetch_adapter_enabled="1"
    ;;
  0|false|FALSE|no|NO)
    prefetch_adapter_enabled="0"
    ;;
  auto|AUTO)
    if [[ "${multi_gpu_enabled}" == "1" ]]; then
      prefetch_adapter_enabled="1"
    fi
    ;;
  *)
    echo "Invalid PREFETCH_ADAPTER value: ${PREFETCH_ADAPTER}. Use auto, 0, or 1."
    exit 4
    ;;
esac

if [[ "${prefetch_enabled}" == "1" && ! -d "${model_ref}" ]]; then
  echo "Prefetch enabled: warming Hugging Face cache for ${model_ref}"
  python - "${model_ref}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
snapshot_download(repo_id=repo_id, resume_download=True)
print(f"Prefetch complete: {repo_id}")
PY
fi

if [[ "${prefetch_adapter_enabled}" == "1" && -n "${adapter_ref}" && ! -d "${adapter_ref}" ]]; then
  if [[ "${adapter_ref}" == /* || "${adapter_ref}" == ./* || "${adapter_ref}" == ../* || "${adapter_ref}" == ~* ]]; then
    echo "Skipping adapter prefetch for unresolved local path: ${adapter_ref}"
  else
    echo "Adapter prefetch enabled: warming Hugging Face cache for ${adapter_ref}"
    python - "${adapter_ref}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
snapshot_download(repo_id=repo_id, resume_download=True)
print(f"Adapter prefetch complete: {repo_id}")
PY
  fi
fi

train_cmd=(finetree-ft-train --config "${CONFIG_PATH}")
if [[ "${multi_gpu_enabled}" == "1" ]]; then
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
  if command -v torchrun >/dev/null 2>&1; then
    train_cmd=(
      torchrun
      --standalone
      --log-dir "${torchrun_log_dir}"
      --redirects 3
      --tee 3
      --nproc_per_node "${NPROC_PER_NODE}"
      --module finetree_annotator.finetune.trainer_unsloth
      --config "${CONFIG_PATH}"
    )
  else
    train_cmd=(
      python -m torch.distributed.run
      --standalone
      --log-dir "${torchrun_log_dir}"
      --redirects 3
      --tee 3
      --nproc_per_node "${NPROC_PER_NODE}"
      --module finetree_annotator.finetune.trainer_unsloth
      --config "${CONFIG_PATH}"
    )
  fi
fi

echo "Starting fine-tune on ${gpu_count} visible GPU(s). Log: ${log_file}"
echo "Launch mode: $([[ "${multi_gpu_enabled}" == "1" ]] && echo "multi-gpu (${NPROC_PER_NODE} proc)" || echo "single-gpu")"
echo "HF_HOME: ${HF_HOME}"
echo "HF_HUB_DISABLE_XET: ${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER: ${HF_HUB_ENABLE_HF_TRANSFER}"
echo "Prefetch model: $([[ "${prefetch_enabled}" == "1" ]] && echo "enabled" || echo "disabled")"
echo "Prefetch adapter: $([[ "${prefetch_adapter_enabled}" == "1" ]] && echo "enabled" || echo "disabled")"
if [[ "${multi_gpu_enabled}" == "1" ]]; then
  echo "Torchrun rank logs: ${torchrun_log_dir}"
fi
echo "Train command: ${train_cmd[*]}"
"${train_cmd[@]}" 2>&1 | tee "${log_file}"

echo "Training finished."
