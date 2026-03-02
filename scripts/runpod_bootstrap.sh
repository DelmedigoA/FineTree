#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/FineTree}"
REPO_URL="${REPO_URL:-https://github.com/DelmedigoA/FineTree.git}"
CONFIG_PATH="${CONFIG_PATH:-configs/finetune_qwen35a3_vl.yaml}"

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
import importlib.util

required = (
    "torch",
    "transformers",
    "datasets",
    "httpx",
    "trl",
    "peft",
    "fastapi",
    "uvicorn",
    "huggingface_hub",
    "unsloth",
    "pytest",
    "yaml",
)
missing = []
for mod in required:
    if importlib.util.find_spec(mod) is None:
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

target_is_qwen35_a3="0"
if [[ -f "${CONFIG_PATH}" ]]; then
  target_is_qwen35_a3="$(
    python - "${CONFIG_PATH}" <<'PY'
import re
import sys
from pathlib import Path

import yaml


def _fallback_is_qwen35_a3(model_id: str) -> bool:
    canonical = re.sub(r"[^a-z0-9]+", "", str(model_id).strip().lower())
    return bool(canonical) and "qwen35" in canonical and "a3" in canonical


cfg_path = Path(sys.argv[1])
raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
model_id = str((raw.get("model") or {}).get("base_model", "")).strip()

is_qwen35_a3 = _fallback_is_qwen35_a3(model_id)
try:
    from finetree_annotator.inference.model_family import is_qwen35_a3_model

    is_qwen35_a3 = is_qwen35_a3_model(model_id)
except Exception:
    pass

print("1" if is_qwen35_a3 else "0")
PY
  )"
fi

stack_needs_repair="$(python - <<'PY'
import contextlib
import io

issues = []

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import unsloth  # noqa: F401
except Exception as exc:  # pragma: no cover
    issues.append(f"unsloth-import:{exc.__class__.__name__}")

print(",".join(issues))
PY
)"

if [[ -n "${stack_needs_repair}" && "${target_is_qwen35_a3}" != "1" ]]; then
  echo "Repairing Unsloth/Transformers stack (${stack_needs_repair})..."
  python -m pip install --upgrade --force-reinstall --no-cache-dir \
    "unsloth" \
    "unsloth_zoo"
elif [[ -n "${stack_needs_repair}" ]]; then
  echo "Skipping generic stack repair for Qwen3.5-A3 target (${stack_needs_repair}); using Qwen3.5 patch path."
fi

if [[ "${target_is_qwen35_a3}" == "1" ]]; then
  qwen35_stack_status="$(python - <<'PY'
import contextlib
import io

issues = []

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "qwen3_5_moe" not in CONFIG_MAPPING:
        issues.append("missing-qwen3_5_moe")
except Exception as exc:  # pragma: no cover
    issues.append(f"transformers-config:{exc.__class__.__name__}")

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import unsloth  # noqa: F401
except Exception as exc:  # pragma: no cover
    issues.append(f"unsloth-import:{exc.__class__.__name__}")

print(",".join(issues))
PY
)"

  if [[ -n "${qwen35_stack_status}" ]]; then
    echo "Applying Qwen3.5-A3 stack patch (${qwen35_stack_status})..."
    python -m pip uninstall -y unsloth unsloth_zoo
    python -m pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
    python -m pip install --no-deps git+https://github.com/unslothai/unsloth.git
    python -m pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo
    python -m pip install --upgrade "transformers==5.2.0"
    python -m pip install --upgrade "torchvision==0.25.0" "bitsandbytes==0.49.2" "xformers==0.0.35"
  else
    echo "Qwen3.5-A3 stack already ready; skipping patch."
  fi
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
     Optional multi-GPU:
       export MULTI_GPU=1
       export NPROC_PER_NODE=<gpu_count>
EOF
