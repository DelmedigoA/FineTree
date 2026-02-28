FROM docker.io/runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv python install 3.12 && uv venv --python 3.12 --seed --system-site-packages /opt/venv

WORKDIR /opt/finetree

# Base app/runtime packages.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "pydantic==2.12.5" \
      "pdf2image==1.17.0" \
      "PyQt5==5.15.11" \
      "google-genai==1.65.0" \
      "PyYAML==6.0.3" \
      "Pillow==12.1.1" \
      "datasets==3.6.0" \
      "peft==0.18.1" \
      "huggingface_hub==1.5.0" \
      "tokenizers==0.22.2" \
      "trl==0.22.2" \
      "transformers==5.2.0"

# Notebook-aligned Qwen3.5 MoE stack:
# - install unsloth / unsloth-zoo from GitHub
# - keep transformers pinned to 5.2.0
RUN python -m pip uninstall -y unsloth unsloth_zoo || true && \
    python -m pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git && \
    python -m pip install --no-deps git+https://github.com/unslothai/unsloth.git && \
    python -m pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo && \
    python -m pip install --upgrade "transformers==5.2.0"

# Fail image build early if Qwen3.5 model type is unavailable.
RUN python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "qwen3_5_moe" not in CONFIG_MAPPING:
    raise SystemExit("qwen3_5_moe not available in CONFIG_MAPPING")
print("Qwen3.5 MoE stack check passed.")
PY

WORKDIR /workspace
CMD ["bash", "-lc", "sleep infinity"]
