#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".env/bin/python" ]]; then
  echo "Missing .env virtualenv. Run bootstrap first." >&2
  exit 1
fi

exec .env/bin/python -m finetree_annotator.deploy.runpod_pod_start "$@"
