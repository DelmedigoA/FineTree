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

echo -e "${YLW}FineTree starting…${RST}"
echo -e "  Backend  → http://localhost:${BACKEND_PORT}"
echo -e "  Frontend → http://localhost:${FRONTEND_PORT}"
echo -e "  Data     → ${ROOT}/${DATA_ROOT}"
echo ""

# Kill children on Ctrl+C
cleanup() {
  echo -e "\n${YLW}Shutting down…${RST}"
  kill 0
}
trap cleanup SIGINT SIGTERM

# Start backend
python -m finetree_annotator.api.app \
  --data-root "${ROOT}/${DATA_ROOT}" \
  --host 127.0.0.1 \
  --port "${BACKEND_PORT}" 2>&1 | \
  sed "s/^/$(printf "${BLU}[backend]${RST} ")/" &

# Wait for backend to be ready
echo -e "${BLU}[backend]${RST} Waiting for API..."
for i in $(seq 1 30); do
  curl -sf "http://localhost:${BACKEND_PORT}/api/health" > /dev/null 2>&1 && break
  sleep 0.5
done

# Start frontend
cd "${ROOT}/web"
npm run dev -- --port "${FRONTEND_PORT}" 2>&1 | \
  sed "s/^/$(printf "${GRN}[frontend]${RST} ")/" &

# Wait for frontend then open browser
sleep 2
echo -e "${GRN}[frontend]${RST} Opening browser..."
open "http://localhost:${FRONTEND_PORT}" 2>/dev/null || \
  xdg-open "http://localhost:${FRONTEND_PORT}" 2>/dev/null || true

# Wait for both
wait
