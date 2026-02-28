#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
REPO_URL="${REPO_URL:-https://github.com/DelmedigoA/FineTree.git}"

if [[ ! -d /workspace ]]; then
  echo "Expected /workspace mount is missing."
  exit 1
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "Cloning repository into ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

if [[ -f /opt/venv/bin/activate ]]; then
  # Preferred path for the custom RunPod image.
  # shellcheck disable=SC1091
  source /opt/venv/bin/activate
else
  # Fallback for non-custom images.
  if [[ ! -d .env ]]; then
    python3 -m venv .env
  fi
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

editable_active="0"
if python - <<'PY'
import importlib.util
from pathlib import Path

repo = Path.cwd().resolve()
spec = importlib.util.find_spec("finetree_annotator")
if spec is None:
    raise SystemExit(1)
origin = getattr(spec, "origin", None)
if not origin:
    raise SystemExit(1)
mod_path = Path(origin).resolve()
if repo in mod_path.parents:
    raise SystemExit(0)
raise SystemExit(1)
PY
then
  editable_active="1"
fi

missing_deps="$(python - <<'PY'
import importlib

required = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "huggingface_hub",
    "unsloth",
    "yaml",
)
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
print(",".join(missing))
PY
)"

if [[ -n "${missing_deps}" ]]; then
  echo "Missing dependencies detected (${missing_deps}). Installing full project dependencies..."
  python -m pip install -e .
elif [[ "${editable_active}" == "1" ]]; then
  echo "Editable install already active; skipping pip install."
else
  python -m pip install -e . --no-deps
fi

cat <<'EOF'
Bootstrap complete.
Next steps:
  1) Export required secrets:
       export FINETREE_HF_TOKEN=...
       export GEMINI_API_KEY=...
  2) Validate and train:
       ./scripts/runpod_validate_data.sh
       ./scripts/runpod_train.sh
EOF
