#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-${ROOT_DIR}/.runpod}"
PID_FILE="${STATE_DIR}/pod-start.pid"
LOG_FILE="${STATE_DIR}/pod-start.log"

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(ts_utc)] [machine-tools] $*"
}

usage() {
  cat <<'EOF'
RunPod machine toolbox for FineTree pod operations over SSH.

Usage:
  scripts/runpod_machine_tools.sh <command> [args]

Commands:
  start [config]        Start finetree-runpod-pod-start in background.
  stop                  Stop background service started by this tool.
  status                Show process/port/health summary.
  check                 Run extended diagnostics (GPU, disk, health, auth).
  flow-check [config]   Validate pod deployment flow/config logic.
  warmup [args...]      Run local warmup against 127.0.0.1:6666.
  logs [n]              Tail service log (default 120 lines).
  logs-find <id> [n]    Show recent log lines around an error id (e.g. poderr-abc123).
  help                  Show this help.

Warmup defaults:
  --base-url http://127.0.0.1:6666
  --api-key from FINETREE_POD_API_KEY
  --attempts 36 --sleep-seconds 10 --timeout-seconds 900
EOF
}

ensure_state_dir() {
  mkdir -p "${STATE_DIR}"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

config_inference_value() {
  local config_path="$1"
  local key="$2"
  awk -v wanted_key="${key}" '
    BEGIN { in_inference = 0 }
    /^inference:[[:space:]]*$/ { in_inference = 1; next }
    in_inference && /^[^[:space:]]/ { in_inference = 0 }
    in_inference {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      if (index(line, wanted_key ":") == 1) {
        value = substr(line, length(wanted_key) + 2)
        sub(/^[[:space:]]+/, "", value)
        sub(/[[:space:]]+#.*$/, "", value)
        gsub(/^"/, "", value)
        gsub(/"$/, "", value)
        gsub(/^'\''/, "", value)
        gsub(/'\''$/, "", value)
        print value
        exit
      }
    }
  ' "${config_path}"
}

is_null_like() {
  local v="${1:-}"
  [[ -z "${v}" || "${v}" == "null" || "${v}" == "None" || "${v}" == "~" ]]
}

flow_check() {
  local config="${1:-${FINETREE_QWEN_CONFIG:-configs/qwen_ui_runpod_pod_local_8bit.yaml}}"
  local errors=0
  local warnings=0

  if [[ ! -f "${config}" ]]; then
    log "FLOW-ERROR missing config file: ${config}"
    return 1
  fi

  local backend endpoint_base_url model_path endpoint_model adapter_path api_key
  backend="$(config_inference_value "${config}" "backend")"
  endpoint_base_url="$(config_inference_value "${config}" "endpoint_base_url")"
  model_path="$(config_inference_value "${config}" "model_path")"
  endpoint_model="$(config_inference_value "${config}" "endpoint_model")"
  adapter_path="$(config_inference_value "${config}" "adapter_path")"
  api_key="${FINETREE_POD_API_KEY:-}"

  log "flow-check config=${config}"
  log "flow-check backend=${backend:-<missing>} model_path=${model_path:-<missing>} endpoint_model=${endpoint_model:-<missing>}"
  log "flow-check endpoint_base_url=${endpoint_base_url:-<missing>} adapter_path=${adapter_path:-<missing>}"

  if [[ -z "${backend}" ]]; then
    log "FLOW-ERROR inference.backend is missing."
    errors=$((errors + 1))
  fi

  if [[ -z "${api_key}" ]]; then
    log "FLOW-ERROR FINETREE_POD_API_KEY is not set."
    errors=$((errors + 1))
  fi

  case "${backend}" in
    local)
      if is_null_like "${model_path}" && [[ -z "${FINETREE_QWEN_MODEL:-}" ]]; then
        log "FLOW-ERROR local backend requires inference.model_path or FINETREE_QWEN_MODEL."
        errors=$((errors + 1))
      fi
      if ! is_null_like "${endpoint_base_url}"; then
        log "FLOW-ERROR local backend should not set inference.endpoint_base_url."
        errors=$((errors + 1))
      fi
      if is_null_like "${adapter_path}" && [[ -z "${FINETREE_ADAPTER_REF:-}" ]]; then
        log "FLOW-WARN no adapter configured (inference.adapter_path and FINETREE_ADAPTER_REF are empty)."
        warnings=$((warnings + 1))
      fi
      ;;
    runpod_openai)
      if is_null_like "${endpoint_base_url}"; then
        log "FLOW-ERROR runpod_openai backend requires inference.endpoint_base_url."
        errors=$((errors + 1))
      fi
      if [[ "${endpoint_base_url}" == *"/chat/completions" ]]; then
        log "FLOW-ERROR endpoint_base_url must be OpenAI base URL ending with /v1, not /chat/completions."
        errors=$((errors + 1))
      fi
      if [[ "${endpoint_base_url}" == *"127.0.0.1:6666"* || "${endpoint_base_url}" == *"localhost:6666"* ]]; then
        log "FLOW-ERROR runpod_openai endpoint points to local pod API (recursive self-call)."
        errors=$((errors + 1))
      fi
      ;;
    runpod_queue)
      if is_null_like "${endpoint_base_url}"; then
        log "FLOW-ERROR runpod_queue backend requires inference.endpoint_base_url."
        errors=$((errors + 1))
      fi
      if [[ "${endpoint_base_url}" == *"/openai/v1"* || "${endpoint_base_url}" == *"/chat/completions" ]]; then
        log "FLOW-ERROR runpod_queue must target queue endpoint base (for /run and /status), not OpenAI URL."
        errors=$((errors + 1))
      fi
      ;;
    *)
      log "FLOW-ERROR unsupported inference.backend=${backend}"
      errors=$((errors + 1))
      ;;
  esac

  if [[ "${errors}" -gt 0 ]]; then
    log "flow-check result=FAIL errors=${errors} warnings=${warnings}"
    return 1
  fi
  log "flow-check result=PASS warnings=${warnings}"
  return 0
}

pid_from_file() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d ' \n\r\t' < "${PID_FILE}"
  fi
}

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

start_service() {
  ensure_state_dir
  local config="${1:-${FINETREE_QWEN_CONFIG:-configs/qwen_ui_runpod_pod_local_8bit.yaml}}"
  local existing_pid
  existing_pid="$(pid_from_file || true)"
  if is_pid_running "${existing_pid}"; then
    log "service already running pid=${existing_pid} log=${LOG_FILE}"
    return 0
  fi

  log "starting service config=${config} log=${LOG_FILE}"
  (
    cd "${ROOT_DIR}"
    nohup finetree-runpod-pod-start --config "${config}" >>"${LOG_FILE}" 2>&1 &
    echo "$!" > "${PID_FILE}"
  )
  sleep 2
  status_service
}

stop_service() {
  local pid
  pid="$(pid_from_file || true)"
  if ! is_pid_running "${pid}"; then
    log "no running service found via ${PID_FILE}"
    rm -f "${PID_FILE}"
    return 0
  fi

  log "stopping pid=${pid}"
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 15); do
    if ! is_pid_running "${pid}"; then
      rm -f "${PID_FILE}"
      log "stopped"
      return 0
    fi
    sleep 1
  done

  log "force-killing pid=${pid}"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}"
}

show_processes() {
  log "processes"
  if have_cmd rg; then
    ps -ef | rg -i "finetree-runpod-pod-start|finetree_annotator.deploy.pod_api|finetree_annotator.deploy.pod_gradio|uvicorn|gradio" || true
  else
    ps -ef | grep -Ei "finetree-runpod-pod-start|finetree_annotator.deploy.pod_api|finetree_annotator.deploy.pod_gradio|uvicorn|gradio" || true
  fi
}

show_ports() {
  log "ports"
  if have_cmd ss; then
    ss -ltnp | (grep -E ":(6666|5555)\b" || true)
  elif have_cmd netstat; then
    netstat -ltn 2>/dev/null | (grep -E "[:.](6666|5555)\b" || true)
  else
    log "neither ss nor netstat is available"
  fi
}

show_health() {
  log "healthz"
  curl -sS -m 5 http://127.0.0.1:6666/healthz || true
  echo
  log "readyz"
  curl -sS -m 5 http://127.0.0.1:6666/readyz || true
  echo
}

show_models_auth() {
  local key="${FINETREE_POD_API_KEY:-}"
  if [[ -z "${key}" ]]; then
    log "FINETREE_POD_API_KEY is not set; skipping /v1/models auth check"
    return 0
  fi
  log "v1/models auth check"
  curl -sS -m 8 \
    -H "Authorization: Bearer ${key}" \
    http://127.0.0.1:6666/v1/models || true
  echo
}

status_service() {
  local pid
  pid="$(pid_from_file || true)"
  if is_pid_running "${pid}"; then
    log "service status=running pid=${pid} log=${LOG_FILE}"
  else
    log "service status=not_running pid_file=${PID_FILE}"
  fi
  show_processes
  show_ports
  show_health
}

check_service() {
  status_service

  log "gpu"
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader || true
  else
    log "nvidia-smi not found"
  fi

  log "disk"
  df -h /workspace 2>/dev/null || true
  df -h /runpod-volume 2>/dev/null || true

  show_models_auth
  flow_check "${FINETREE_QWEN_CONFIG:-configs/qwen_ui_runpod_pod_local_8bit.yaml}"
}

warmup_local() {
  local key="${FINETREE_POD_API_KEY:-}"
  if [[ -z "${key}" ]]; then
    log "FINETREE_POD_API_KEY is required for warmup"
    return 1
  fi

  local warmup_script="${ROOT_DIR}/scripts/runpod_api_warmup.sh"
  if [[ ! -x "${warmup_script}" ]]; then
    log "warmup script is missing or not executable: ${warmup_script}"
    return 1
  fi

  log "running warmup on 127.0.0.1:6666"
  "${warmup_script}" \
    --base-url "http://127.0.0.1:6666" \
    --api-key "${key}" \
    --max-tokens 10 \
    --attempts 36 \
    --sleep-seconds 10 \
    --timeout-seconds 900 \
    "$@"
}

tail_logs() {
  local n="${1:-120}"
  ensure_state_dir
  if [[ ! -f "${LOG_FILE}" ]]; then
    log "log file does not exist yet: ${LOG_FILE}"
    return 1
  fi
  tail -n "${n}" -f "${LOG_FILE}"
}

find_log_error() {
  local err_id="${1:-}"
  local n="${2:-200}"
  ensure_state_dir
  if [[ -z "${err_id}" ]]; then
    log "usage: logs-find <error_id> [n]"
    return 1
  fi
  if [[ ! -f "${LOG_FILE}" ]]; then
    log "log file does not exist yet: ${LOG_FILE}"
    return 1
  fi

  log "searching log for error_id=${err_id}"
  if have_cmd rg; then
    rg -n -C 25 "${err_id}" "${LOG_FILE}" || true
  else
    grep -n "${err_id}" "${LOG_FILE}" || true
  fi
  log "last ${n} log lines"
  tail -n "${n}" "${LOG_FILE}" || true
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    start) start_service "$@" ;;
    stop) stop_service ;;
    status) status_service ;;
    check) check_service ;;
    flow-check) flow_check "$@" ;;
    warmup) warmup_local "$@" ;;
    logs) tail_logs "$@" ;;
    logs-find) find_log_error "$@" ;;
    help|-h|--help) usage ;;
    *)
      log "unknown command: ${cmd}"
      usage
      return 1
      ;;
  esac
}

main "$@"
