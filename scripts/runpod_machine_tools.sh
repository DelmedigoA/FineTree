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
  warmup [args...]      Run local warmup against 127.0.0.1:6666.
  logs [n]              Tail service log (default 120 lines).
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

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    start) start_service "$@" ;;
    stop) stop_service ;;
    status) status_service ;;
    check) check_service ;;
    warmup) warmup_local "$@" ;;
    logs) tail_logs "$@" ;;
    help|-h|--help) usage ;;
    *)
      log "unknown command: ${cmd}"
      usage
      return 1
      ;;
  esac
}

main "$@"
