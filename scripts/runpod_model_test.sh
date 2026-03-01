#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(ts_utc)] [model-test] $*"
}

CONFIG_PATH="${1:-${FINETREE_QWEN_CONFIG:-${ROOT_DIR}/configs/qwen_ui_runpod_pod_local_8bit.yaml}}"
PROMPT_TEXT="${2:-${FINETREE_MODEL_TEST_PROMPT:-Reply exactly: warmup-ok}}"
MAX_TOKENS="${3:-${FINETREE_MODEL_TEST_MAX_TOKENS:-8}}"
IMAGE_PATH="${FINETREE_MODEL_TEST_IMAGE_PATH:-/tmp/ft_smoke.png}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  log "missing config: ${CONFIG_PATH}"
  exit 1
fi

if ! [[ "${MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  log "max tokens must be numeric, got: ${MAX_TOKENS}"
  exit 1
fi

log "config=${CONFIG_PATH}"
log "max_tokens=${MAX_TOKENS}"

ADAPTER_REF_OVERRIDE="${FINETREE_MODEL_TEST_ADAPTER_REF:-}"
QUANT_OVERRIDE="${FINETREE_MODEL_TEST_QUANTIZATION:-}"
LOAD_IN_4BIT_OVERRIDE="${FINETREE_MODEL_TEST_LOAD_IN_4BIT:-}"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
FINETREE_ADAPTER_REF="${ADAPTER_REF_OVERRIDE}" \
FINETREE_QWEN_QUANTIZATION="${QUANT_OVERRIDE}" \
FINETREE_QWEN_LOAD_IN_4BIT="${LOAD_IN_4BIT_OVERRIDE}" \
python - "${CONFIG_PATH}" "${PROMPT_TEXT}" "${MAX_TOKENS}" "${IMAGE_PATH}" <<'PY'
import base64
import sys
from pathlib import Path

from finetree_annotator.qwen_vlm import generate_content_from_image

config_path = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
image_path = Path(sys.argv[4])

image_path.write_bytes(
    base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6p6nQAAAAASUVORK5CYII=")
)

try:
    import torch
except Exception:
    torch = None

if torch is not None:
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu_name={torch.cuda.get_device_name(0)}")
        print(f"cuda_mem_before_mb={torch.cuda.memory_allocated(0) // (1024 * 1024)}")

output = generate_content_from_image(
    image_path=image_path,
    prompt=prompt,
    config_path=config_path,
    max_new_tokens=max_tokens,
)
print(output)

if torch is not None and torch.cuda.is_available():
    print(f"cuda_mem_after_mb={torch.cuda.memory_allocated(0) // (1024 * 1024)}")
    print(f"cuda_reserved_after_mb={torch.cuda.memory_reserved(0) // (1024 * 1024)}")
PY
