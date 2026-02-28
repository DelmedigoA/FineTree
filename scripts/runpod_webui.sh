#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1234}"
IMAGES_DIR="${IMAGES_DIR:-data/pdf_images/test}"
ANNOTATIONS_PATH="${ANNOTATIONS_PATH:-}"
PROMPT_PATH="${PROMPT_PATH:-prompt.txt}"

cd "${REPO_DIR}"

if [[ -f /opt/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /opt/venv/bin/activate
elif [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

python3 -m pip install -e . --no-build-isolation

cmd=(
  finetree-web-ui
  --host "${HOST}"
  --port "${PORT}"
  --images-dir "${IMAGES_DIR}"
  --prompt-path "${PROMPT_PATH}"
)

if [[ -n "${ANNOTATIONS_PATH}" ]]; then
  cmd+=(--annotations "${ANNOTATIONS_PATH}")
fi

echo "Starting FineTree Web UI on ${HOST}:${PORT}"
echo "Command: ${cmd[*]}"
"${cmd[@]}"
