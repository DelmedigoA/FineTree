#!/bin/bash
# FineTree — start backend + frontend together

DATA_ROOT="${1:-data}"
BACKEND_PORT="${2:-8000}"
FRONTEND_PORT="${3:-5173}"

# Colors
GRN='\033[0;32m'
BLU='\033[0;34m'
YLW='\033[1;33m'
RST='\033[0m'

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

# Fallback if no venv
if [ ! -f "$PYTHON" ]; then
  PYTHON="$(which python3)"
fi

echo -e "${YLW}FineTree starting…${RST}"
echo -e "  Backend  → http://localhost:${BACKEND_PORT}"
echo -e "  Frontend → http://localhost:${FRONTEND_PORT}"
echo -e "  Data     → ${ROOT}/${DATA_ROOT}"
echo -e "  Python   → ${PYTHON}"
echo ""

# Kill any stale processes on those ports
for PORT in "${BACKEND_PORT}" "${FRONTEND_PORT}"; do
  PID=$(lsof -ti tcp:"$PORT" 2>/dev/null)
  if [ -n "$PID" ]; then
    echo -e "${YLW}Killing stale process on port ${PORT} (PID ${PID})${RST}"
    kill -9 $PID 2>/dev/null
  fi
done
sleep 0.5

# Kill children on Ctrl+C
cleanup() {
  echo -e "\n${YLW}Shutting down…${RST}"
  kill -- -$$ 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
"$PYTHON" -m finetree_annotator.api.app \
  --data-root "${ROOT}/${DATA_ROOT}" \
  --host 127.0.0.1 \
  --port "${BACKEND_PORT}" 2>&1 | \
  sed "s/^/[backend] /" &
BACKEND_PID=$!

# Wait for backend to be ready (up to 15s)
echo "[backend] Waiting for API..."
for i in $(seq 1 30); do
  curl -sf "http://localhost:${BACKEND_PORT}/api/health" > /dev/null 2>&1 && break
  sleep 0.5
done

if ! curl -sf "http://localhost:${BACKEND_PORT}/api/health" > /dev/null 2>&1; then
  echo "[backend] WARNING: API didn't respond, starting frontend anyway…"
fi

# Start frontend
cd "${ROOT}/web"
export PATH="/opt/homebrew/Cellar/node/24.9.0/bin:$PATH"
npm run dev -- --port "${FRONTEND_PORT}" 2>&1 | sed "s/^/[frontend] /" &
FRONTEND_PID=$!

# Open browser after a short wait
sleep 2
echo "[frontend] Opening browser → http://localhost:${FRONTEND_PORT}"
open "http://localhost:${FRONTEND_PORT}" 2>/dev/null || \
  xdg-open "http://localhost:${FRONTEND_PORT}" 2>/dev/null || true

wait
