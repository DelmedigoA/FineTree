#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(ts_utc)] [deploy-doppler] $*"
}

usage() {
  cat <<'EOF'
Deploy FineTree pod image and env using Doppler secrets, then run warmup.

Usage:
  scripts/runpod_deploy_doppler.sh [options]

Options:
  --project <name>       Doppler project (default: DOPPLER_PROJECT or current scope)
  --config <name>        Doppler config (default: DOPPLER_CONFIG or current scope)
  --pod-id <id>          RunPod pod id (overrides Doppler secret POD_ID/RUNPOD_POD_ID)
  --image-name <name>    Image repo (overrides FINETREE_POD_IMAGE_NAME/IMAGE_NAME secret)
  --image-tag <tag>      Image tag (overrides FINETREE_POD_IMAGE_TAG/IMAGE_TAG secret, default: latest)
  --qwen-config <path>   FINETREE_QWEN_CONFIG value (default: configs/qwen_ui_runpod_pod_local_8bit.yaml)
  --skip-build           Skip docker buildx --push
  --skip-update          Skip runpodctl pod update
  --skip-warmup          Skip warmup call after update
  -h, --help             Show this help

Required secrets in Doppler (minimum):
  FINETREE_POD_API_KEY
  POD_ID (or RUNPOD_POD_ID) unless --pod-id is provided
  FINETREE_POD_IMAGE_NAME (or IMAGE_NAME) unless --image-name is provided

Optional secrets:
  FINETREE_POD_IMAGE_TAG or IMAGE_TAG
  FINETREE_GRADIO_USER
  FINETREE_GRADIO_PASS
  FINETREE_MAX_CONCURRENCY
  FINETREE_SERVED_MODEL_NAME
  FINETREE_ADAPTER_REF
  FINETREE_QWEN_QUANTIZATION
  FINETREE_POD_DEBUG_ERRORS
  HUGGING_FACE_HUB_TOKEN
  HF_TOKEN
EOF
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

require_cmd() {
  local cmd="$1"
  if ! have_cmd "${cmd}"; then
    log "missing required command: ${cmd}"
    exit 1
  fi
}

PROJECT="${DOPPLER_PROJECT:-}"
CONFIG="${DOPPLER_CONFIG:-}"
POD_ID_OVERRIDE=""
IMAGE_NAME_OVERRIDE=""
IMAGE_TAG_OVERRIDE=""
QWEN_CONFIG_PATH="configs/qwen_ui_runpod_pod_local_8bit.yaml"
SKIP_BUILD=0
SKIP_UPDATE=0
SKIP_WARMUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --pod-id)
      POD_ID_OVERRIDE="${2:-}"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME_OVERRIDE="${2:-}"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG_OVERRIDE="${2:-}"
      shift 2
      ;;
    --qwen-config)
      QWEN_CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-update)
      SKIP_UPDATE=1
      shift
      ;;
    --skip-warmup)
      SKIP_WARMUP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

require_cmd doppler
require_cmd jq
require_cmd runpodctl

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  require_cmd docker
fi

doppler_args=()
if [[ -n "${PROJECT}" ]]; then
  doppler_args+=(--project "${PROJECT}")
fi
if [[ -n "${CONFIG}" ]]; then
  doppler_args+=(--config "${CONFIG}")
fi

log "loading secrets from Doppler"
SECRETS_JSON="$(doppler secrets download --format json --no-file "${doppler_args[@]}")"

secret_value() {
  local key="$1"
  jq -r --arg key "${key}" '.[$key] // ""' <<< "${SECRETS_JSON}"
}

first_non_empty() {
  local value=""
  for candidate in "$@"; do
    if [[ -n "${candidate}" ]]; then
      value="${candidate}"
      break
    fi
  done
  echo "${value}"
}

POD_ID="$(first_non_empty "${POD_ID_OVERRIDE}" "$(secret_value POD_ID)" "$(secret_value RUNPOD_POD_ID)")"
IMAGE_NAME="$(first_non_empty "${IMAGE_NAME_OVERRIDE}" "$(secret_value FINETREE_POD_IMAGE_NAME)" "$(secret_value IMAGE_NAME)")"
IMAGE_TAG="$(first_non_empty "${IMAGE_TAG_OVERRIDE}" "$(secret_value FINETREE_POD_IMAGE_TAG)" "$(secret_value IMAGE_TAG)" "latest")"
FINETREE_POD_API_KEY="$(secret_value FINETREE_POD_API_KEY)"
FINETREE_GRADIO_USER="$(secret_value FINETREE_GRADIO_USER)"
FINETREE_GRADIO_PASS="$(secret_value FINETREE_GRADIO_PASS)"
FINETREE_MAX_CONCURRENCY="$(secret_value FINETREE_MAX_CONCURRENCY)"
FINETREE_SERVED_MODEL_NAME="$(secret_value FINETREE_SERVED_MODEL_NAME)"
FINETREE_ADAPTER_REF="$(secret_value FINETREE_ADAPTER_REF)"
FINETREE_QWEN_QUANTIZATION="$(secret_value FINETREE_QWEN_QUANTIZATION)"
FINETREE_POD_DEBUG_ERRORS="$(first_non_empty "$(secret_value FINETREE_POD_DEBUG_ERRORS)" "0")"
HUGGING_FACE_HUB_TOKEN="$(secret_value HUGGING_FACE_HUB_TOKEN)"
HF_TOKEN="$(secret_value HF_TOKEN)"

if [[ -z "${POD_ID}" ]]; then
  log "missing pod id. set --pod-id or Doppler secret POD_ID/RUNPOD_POD_ID"
  exit 1
fi
if [[ -z "${IMAGE_NAME}" ]]; then
  log "missing image name. set --image-name or Doppler secret FINETREE_POD_IMAGE_NAME/IMAGE_NAME"
  exit 1
fi
if [[ -z "${FINETREE_POD_API_KEY}" ]]; then
  log "missing FINETREE_POD_API_KEY in Doppler"
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${QWEN_CONFIG_PATH}" ]]; then
  log "qwen config file not found: ${QWEN_CONFIG_PATH}"
  exit 1
fi

log "flow-check before deploy"
(
  cd "${ROOT_DIR}"
  FINETREE_POD_API_KEY="${FINETREE_POD_API_KEY}" \
  FINETREE_ADAPTER_REF="${FINETREE_ADAPTER_REF}" \
  FINETREE_QWEN_CONFIG="${QWEN_CONFIG_PATH}" \
  scripts/runpod_machine_tools.sh flow-check "${QWEN_CONFIG_PATH}"
)

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
log "target pod=${POD_ID} image=${FULL_IMAGE}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  log "building and pushing image ${FULL_IMAGE}"
  (
    cd "${ROOT_DIR}"
    docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile.pod -t "${FULL_IMAGE}" --push .
  )
else
  log "skipping build"
fi

env_json="$(
  jq -n \
    --arg FINETREE_POD_API_KEY "${FINETREE_POD_API_KEY}" \
    --arg FINETREE_QWEN_CONFIG "${QWEN_CONFIG_PATH}" \
    --arg FINETREE_POD_DEBUG_ERRORS "${FINETREE_POD_DEBUG_ERRORS}" \
    --arg FINETREE_GRADIO_USER "${FINETREE_GRADIO_USER}" \
    --arg FINETREE_GRADIO_PASS "${FINETREE_GRADIO_PASS}" \
    --arg FINETREE_MAX_CONCURRENCY "${FINETREE_MAX_CONCURRENCY}" \
    --arg FINETREE_SERVED_MODEL_NAME "${FINETREE_SERVED_MODEL_NAME}" \
    --arg FINETREE_ADAPTER_REF "${FINETREE_ADAPTER_REF}" \
    --arg FINETREE_QWEN_QUANTIZATION "${FINETREE_QWEN_QUANTIZATION}" \
    --arg HUGGING_FACE_HUB_TOKEN "${HUGGING_FACE_HUB_TOKEN}" \
    --arg HF_TOKEN "${HF_TOKEN}" \
    '
      {
        FINETREE_POD_API_KEY: $FINETREE_POD_API_KEY,
        FINETREE_QWEN_CONFIG: $FINETREE_QWEN_CONFIG,
        FINETREE_POD_DEBUG_ERRORS: $FINETREE_POD_DEBUG_ERRORS
      }
      + (if $FINETREE_GRADIO_USER != "" then {FINETREE_GRADIO_USER: $FINETREE_GRADIO_USER} else {} end)
      + (if $FINETREE_GRADIO_PASS != "" then {FINETREE_GRADIO_PASS: $FINETREE_GRADIO_PASS} else {} end)
      + (if $FINETREE_MAX_CONCURRENCY != "" then {FINETREE_MAX_CONCURRENCY: $FINETREE_MAX_CONCURRENCY} else {} end)
      + (if $FINETREE_SERVED_MODEL_NAME != "" then {FINETREE_SERVED_MODEL_NAME: $FINETREE_SERVED_MODEL_NAME} else {} end)
      + (if $FINETREE_ADAPTER_REF != "" then {FINETREE_ADAPTER_REF: $FINETREE_ADAPTER_REF} else {} end)
      + (if $FINETREE_QWEN_QUANTIZATION != "" then {FINETREE_QWEN_QUANTIZATION: $FINETREE_QWEN_QUANTIZATION} else {} end)
      + (if $HUGGING_FACE_HUB_TOKEN != "" then {HUGGING_FACE_HUB_TOKEN: $HUGGING_FACE_HUB_TOKEN} else {} end)
      + (if $HF_TOKEN != "" then {HF_TOKEN: $HF_TOKEN} else {} end)
    '
)"

if [[ "${SKIP_UPDATE}" -eq 0 ]]; then
  log "updating pod image/env"
  runpodctl pod update "${POD_ID}" --image "${FULL_IMAGE}" --env "${env_json}" >/dev/null
else
  log "skipping pod update"
fi

if [[ "${SKIP_WARMUP}" -eq 0 ]]; then
  log "running warmup against pod proxy"
  (
    cd "${ROOT_DIR}"
    FINETREE_POD_API_KEY="${FINETREE_POD_API_KEY}" \
    scripts/runpod_api_warmup.sh \
      --pod-id "${POD_ID}" \
      --api-key "${FINETREE_POD_API_KEY}" \
      --max-tokens 10 \
      --attempts 24 \
      --sleep-seconds 10 \
      --timeout-seconds 180
  )
else
  log "skipping warmup"
fi

log "deploy completed"
