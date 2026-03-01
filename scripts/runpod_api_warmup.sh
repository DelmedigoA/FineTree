#!/usr/bin/env bash
set -euo pipefail

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_info() {
  echo "[$(ts_utc)] [warmup] $*"
}

usage() {
  cat <<'EOF'
Warm up a FineTree pod API with a tiny multimodal request.

Usage:
  runpod_api_warmup.sh [options]

Options:
  --pod-id <id>           RunPod pod id (used to build https://<id>-6666.proxy.runpod.net).
  --base-url <url>        Override API base URL (example: http://127.0.0.1:6666).
  --api-key <token>       Pod API bearer token. Defaults to FINETREE_POD_API_KEY.
  --model <name>          Model name sent to /v1/chat/completions (default: qwen-gt).
  --max-tokens <n>        Max generated tokens for warmup request (default: 10).
  --attempts <n>          Number of warmup retries (default: 18).
  --sleep-seconds <n>     Delay between retries in seconds (default: 15).
  --timeout-seconds <n>   Per-request timeout in seconds (default: 120).
  --prompt <text>         Warmup prompt text.
  -h, --help              Show this help.

Environment fallbacks:
  POD_ID, BASE_URL, FINETREE_POD_API_KEY, MODEL, MAX_TOKENS, ATTEMPTS,
  SLEEP_SECONDS, TIMEOUT_SECONDS, PROMPT, TINY_PNG_B64.
EOF
}

POD_ID="${POD_ID:-}"
BASE_URL="${BASE_URL:-}"
API_KEY="${FINETREE_POD_API_KEY:-}"
MODEL="${MODEL:-qwen-gt}"
MAX_TOKENS="${MAX_TOKENS:-10}"
ATTEMPTS="${ATTEMPTS:-18}"
SLEEP_SECONDS="${SLEEP_SECONDS:-15}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
PROMPT="${PROMPT:-Reply exactly: warmup-ok}"
TINY_PNG_B64="${TINY_PNG_B64:-iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAACXBIWXMAAAABAAAAAQBPJcTWAAAAYUlEQVR4nO3PwQkAIBDAMAX33/jAIXwEoZmg3TOzfnZ0wKsGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAsxmwP1XYHiXwAAAABJRU5ErkJggg==}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pod-id)
      POD_ID="${2:-}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:-}"
      shift 2
      ;;
    --attempts)
      ATTEMPTS="${2:-}"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="${2:-}"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BASE_URL}" ]]; then
  if [[ -z "${POD_ID}" ]]; then
    echo "Missing pod target. Set --pod-id (or POD_ID) or --base-url (or BASE_URL)." >&2
    exit 1
  fi
  BASE_URL="https://${POD_ID}-6666.proxy.runpod.net"
fi
BASE_URL="${BASE_URL%/}"

if [[ -z "${API_KEY}" ]]; then
  echo "Missing API key. Set --api-key or FINETREE_POD_API_KEY." >&2
  exit 1
fi

for number_var in MAX_TOKENS ATTEMPTS SLEEP_SECONDS TIMEOUT_SECONDS; do
  value="${!number_var}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${number_var} must be a non-negative integer, got: ${value}" >&2
    exit 1
  fi
done

IMAGE_DATA_URI="data:image/png;base64,${TINY_PNG_B64}"
HEALTH_URL="${BASE_URL}/healthz"
CHAT_URL="${BASE_URL}/v1/chat/completions"

started_epoch="$(date +%s)"
log_info "started target=${CHAT_URL}"
log_info "config model=${MODEL} max_tokens=${MAX_TOKENS} attempts=${ATTEMPTS} sleep=${SLEEP_SECONDS}s timeout=${TIMEOUT_SECONDS}s"

for attempt in $(seq 1 "${ATTEMPTS}"); do
  attempt_epoch="$(date +%s)"
  log_info "attempt=${attempt}/${ATTEMPTS} phase=healthz begin url=${HEALTH_URL}"
  health_tmp="$(mktemp)"
  health_code="$(curl -sS -m 20 -o "${health_tmp}" -w "%{http_code}" "${HEALTH_URL}" || true)"
  if [[ "${health_code}" != "200" ]] || ! jq -e '.ok == true' "${health_tmp}" >/dev/null 2>&1; then
    health_preview="$(tr '\n' ' ' < "${health_tmp}" | cut -c1-200 || true)"
    rm -f "${health_tmp}"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=healthz not_ready http=${health_code} body='${health_preview}'"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=sleep seconds=${SLEEP_SECONDS}"
    sleep "${SLEEP_SECONDS}"
    continue
  fi
  rm -f "${health_tmp}"
  log_info "attempt=${attempt}/${ATTEMPTS} phase=healthz ready"

  payload="$(
    jq -n \
      --arg model "${MODEL}" \
      --arg prompt "${PROMPT}" \
      --arg image_url "${IMAGE_DATA_URI}" \
      --arg max_tokens "${MAX_TOKENS}" \
      '{
        model: $model,
        stream: false,
        max_tokens: ($max_tokens | tonumber),
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: $prompt },
              { type: "image_url", image_url: { url: $image_url } }
            ]
          }
        ]
      }'
  )"

  log_info "attempt=${attempt}/${ATTEMPTS} phase=request begin url=${CHAT_URL}"
  set +e
  response="$(curl -sS -m "${TIMEOUT_SECONDS}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${payload}" \
    "${CHAT_URL}" 2>&1)"
  curl_code=$?
  set -e

  if [[ "${curl_code}" -ne 0 ]]; then
    elapsed="$(( $(date +%s) - attempt_epoch ))"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=request failed curl_code=${curl_code} elapsed=${elapsed}s msg='${response}'"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=sleep seconds=${SLEEP_SECONDS}"
    sleep "${SLEEP_SECONDS}"
    continue
  fi

  elapsed="$(( $(date +%s) - attempt_epoch ))"
  log_info "attempt=${attempt}/${ATTEMPTS} phase=request completed elapsed=${elapsed}s"

  content="$(jq -r '.choices[0].message.content // empty' <<< "${response}" 2>/dev/null || true)"
  if [[ -n "${content}" ]]; then
    total_elapsed="$(( $(date +%s) - started_epoch ))"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=done status=success total_elapsed=${total_elapsed}s"
    log_info "response='${content}'"
    exit 0
  fi

  error_text="$(jq -r '.error.message // .detail // empty' <<< "${response}" 2>/dev/null || true)"
  if [[ -n "${error_text}" ]]; then
    log_info "attempt=${attempt}/${ATTEMPTS} phase=request loading detail='${error_text}'"
  else
    response_preview="$(tr '\n' ' ' <<< "${response}" | cut -c1-250)"
    log_info "attempt=${attempt}/${ATTEMPTS} phase=request non_ready body='${response_preview}'"
  fi

  log_info "attempt=${attempt}/${ATTEMPTS} phase=sleep seconds=${SLEEP_SECONDS}"
  sleep "${SLEEP_SECONDS}"
done

total_elapsed="$(( $(date +%s) - started_epoch ))"
log_info "status=failed attempts=${ATTEMPTS} total_elapsed=${total_elapsed}s"
exit 1
