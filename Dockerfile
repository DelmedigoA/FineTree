FROM docker.io/runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv python install 3.12 && uv venv --python 3.12 --seed /opt/venv

WORKDIR /opt/finetree

# Install a pinned stack to avoid long pip backtracking during every pod startup.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "pydantic==2.12.5" \
      "pdf2image==1.17.0" \
      "PyQt5==5.15.11" \
      "google-genai==1.65.0" \
      "PyYAML==6.0.3" \
      "Pillow==12.1.1" \
      "transformers==5.2.0" \
      "datasets==4.6.0" \
      "trl==0.22.2" \
      "peft==0.18.1" \
      "huggingface_hub==0.36.2" \
      "tokenizers" \
      "unsloth" \
      "unsloth_zoo"

WORKDIR /workspace
CMD ["bash", "-lc", "sleep infinity"]
