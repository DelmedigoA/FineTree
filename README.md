# FineTree

## PDF Annotator (PyQt5)

This repo now includes a simple local annotation app:

- Draw bounding boxes over page images.
- Double-click inside a box to edit full `Fact` fields.
- Duplicate selected boxes to avoid retyping repeated facts.
- Fill per-page metadata (`PageMeta`) on the right panel.

### Install

```bash
pip install -r requirements.txt
pip install -e .
```

If your environment is offline:

```bash
pip install -e . --no-build-isolation
```

### Run

```bash
finetree-annotator --images-dir data/pdf_images/test
```

Alternative module form:

```bash
python -m finetree_annotator --images-dir data/pdf_images/test
```

Optional output path:

```bash
finetree-annotator --images-dir data/pdf_images/test --annotations data/annotations/test.json
```

### Source layout

- `src/finetree_annotator/` contains application source code.
- CLI entrypoints are defined in `pyproject.toml`.

### PDF Conversion

```bash
finetree-pdf-to-images data/raw_pdfs/test.pdf
```

### Gemini VLM (Image + Prompt)

```bash
export GOOGLE_API_KEY=your_key_here
finetree-gemini-vlm path/to/small-sample.jpg "Caption this image."
```

Alternative (local config file):

```toml
# finetree_config.toml
api_key = "your_key_here"
```

Or set a custom config location:

```bash
export FINETREE_CONFIG_PATH=/absolute/path/to/finetree_config.toml
```

Optional flags:

- `--model` (default: `gemini-3-flash-preview`)
- `--mime-type` (override auto-detected image MIME type)
- `--api-key` (use explicit key instead of env vars)

### Qwen VLM (Local, Multimodal)

`Qwen GT` uses local model inference and requires CUDA.
The loader supports any Qwen 3.5 A3-family model id (not only a single repo name),
and `inference.adapter_path` can be either a local folder or a Hugging Face repo id.

```bash
finetree-qwen-gt --config configs/finetune_qwen35a3_vl.yaml --image data/pdf_images/test/page_0001.png --stream
```

Optional:

- `--prompt` inline prompt text
- `--prompt-path` prompt file path (defaults to `prompt.txt`)
- `--model` override model id/path for inference

### Fine-Tuning Pipeline (Unsloth)

Default config: `configs/finetune_qwen35a3_vl.yaml`

Build train/val datasets:

```bash
finetree-ft-build-dataset --config configs/finetune_qwen35a3_vl.yaml
```

Train adapters:

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml
```

Push existing adapters to Hub (without retraining):

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --push-adapter-only
```

Validate environment/config without training:

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --dry-run
```

Optional merge after training:

```bash
finetree-ft-merge-push --config configs/finetune_qwen35a3_vl.yaml
```

Build + export + push dataset JSONL/images to Hugging Face Dataset Hub:

```bash
finetree-ft-push-dataset --config configs/finetune_qwen35a3_vl.yaml --repo-id <user>/<dataset-repo>
```

### RunPod GPU Validation + Full Fine-Tune

Recommended workflow on a fresh RunPod pod:

```bash
cd /workspace
git clone <YOUR_REPO_URL> FineTree
cd FineTree
./scripts/runpod_bootstrap.sh
```

Single-command smoke test (bootstrap + preflight + trainer dry-run):

```bash
./scripts/runpod_smoke_test.sh
```

If you use this as pod start command, use:

```bash
bash -lc "cd /workspace/FineTree && ./scripts/runpod_bootstrap.sh && sleep infinity"
```

`sleep infinity` prevents RunPod from repeatedly restarting the container command and re-running install steps.
`runpod_bootstrap.sh` auto-repairs broken Unsloth installs when imports fail.
If your config targets a Qwen3.5-A3 model family id, it also applies the notebook-style Qwen3.5 patch
(installs `unsloth` + `unsloth-zoo` from GitHub and upgrades `transformers` to `5.2.0` when needed).

Validate environment + data:

```bash
./scripts/runpod_validate_data.sh
```

Or run preflight directly (adds HF auth check):

```bash
finetree-ft-preflight --config configs/finetune_qwen35a3_vl.yaml --check-hf
```

Set secrets:

```bash
export FINETREE_HF_TOKEN=<your_hf_token>
export GEMINI_API_KEY=<your_gemini_key>
```

Build dataset + train in one command:

```bash
./scripts/runpod_train.sh
```

Multi-GPU launch (RunPod with 2+ GPUs):

```bash
export MULTI_GPU=1
# Optional override. Defaults to detected GPU count.
export NPROC_PER_NODE=2
# Optional cache warmup for remote adapter repo in inference.adapter_path.
export PREFETCH_ADAPTER=1
./scripts/runpod_train.sh
```

Launch modes for `runpod_train.sh`:

- `MULTI_GPU=auto` (default): uses distributed launch when more than 1 GPU is visible.
- `MULTI_GPU=1`: force distributed launch, fail fast if fewer than 2 GPUs are available.
- `MULTI_GPU=0`: force single-process launch.

Distributed runs use `torchrun --standalone --nproc_per_node ...`.
Use `training.ddp_find_unused_parameters: false` (default) unless your adapter/module setup requires otherwise.

Or explicit smoke path:

```bash
./scripts/runpod_smoke_test.sh
```

Optional manual steps:

```bash
finetree-ft-build-dataset --config configs/finetune_qwen35a3_vl.yaml
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --dry-run
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml
```

### RunPod Endpoint Prep

Generate endpoint env values directly from your FineTree config:

```bash
finetree-runpod-endpoint-env \
  --config configs/finetune_qwen35a3_vl.yaml \
  --served-model-name qwenasaf \
  --output artifacts/runpod/endpoint.env
```

This creates a sorted `.env`-style file with values like:

- `MODEL_NAME`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `OPENAI_SERVED_MODEL_NAME_OVERRIDE` (if provided)
- `FINETREE_ADAPTER_REF` (when `inference.adapter_path` is set)

To use PyQt Qwen GT via RunPod OpenAI-compatible endpoint, set in your config:

```yaml
inference:
  backend: runpod_openai
  endpoint_base_url: https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
  endpoint_api_key_env: RUNPOD_API_KEY
  endpoint_model: qwenasaf
```

To use PyQt Qwen GT via RunPod Serverless queue endpoint (`/run` + `/status`), set:

```yaml
inference:
  backend: runpod_queue
  endpoint_base_url: https://api.runpod.ai/v2/<ENDPOINT_ID>
  endpoint_api_key_env: RUNPOD_API_KEY
  endpoint_model: Qwen/Qwen3.5-35B-A3B
  endpoint_timeout_sec: 600
```

Template for original (non-finetuned) Qwen:

- `configs/qwen_ui_runpod_queue_qwen35_original.template.yaml`

Optional override when status polling is on a different URL:

```yaml
inference:
  endpoint_status_base_url: https://api.runpod.ai/v2/<ENDPOINT_ID>/status
```

Enable Hub push in config only when needed:

```bash
# push_to_hub.enabled: true
# push_to_hub.repo_id: <your_org>/<your_model_repo>
# keep push_to_hub.hf_token: null (use FINETREE_HF_TOKEN env var)
```

Push adapters after training with current config:

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --push-adapter-only
```

### RunPod Serverless Queue Worker

This repo now includes a queue-style RunPod Serverless worker implementation:

- module: `src/finetree_annotator/deploy/runpod_serverless_worker.py`
- CLI entrypoint: `finetree-runpod-worker`
- sample payload: `deploy/runpod/test_input.json`
- worker is configured for streaming (`yield`) with `return_aggregate_stream=true`

Local payload test:

```bash
finetree-runpod-worker --test-input deploy/runpod/test_input.json --pretty
```

Run as Serverless worker process:

```bash
finetree-runpod-worker --serve
```

Recommended RunPod endpoint container command:

```bash
finetree-runpod-worker --serve
```

For true token streaming in UI, the endpoint must expose both:

- `POST /run` (submit job)
- `GET /stream/{job_id}` (stream events)

The UI queue backend uses `/stream/{job_id}` first and falls back to `/status/{job_id}` when streaming is unavailable.

Build and push a Serverless worker image:

```bash
export IMAGE_NAME=<registry>/<namespace>/finetree-serverless
export IMAGE_TAG=latest
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile.serverless -t "${IMAGE_NAME}:${IMAGE_TAG}" --push .
```

### RunPod Pod Services (Additive to Serverless)

Serverless queue flow remains supported and unchanged. Pod services are an additional path.

Pod Qwen config templates:

- `configs/qwen_ui_runpod_pod_openai.yaml` (proxy mode to external OpenAI-compatible endpoint)
- `configs/qwen_ui_runpod_pod_local_8bit.yaml` (local model loading, 8-bit quantized)

Start both Pod services in one process group:

```bash
finetree-runpod-pod-start --config configs/qwen_ui_runpod_pod_local_8bit.yaml
```

This starts:

- API on `6666` (`/v1/chat/completions`, supports streaming)
- Gradio test app on `5555`

Required Pod auth env vars:

```bash
export FINETREE_POD_API_KEY=<token-for-6666>
export FINETREE_GRADIO_USER=<gradio-user>
export FINETREE_GRADIO_PASS=<gradio-pass>
export FINETREE_ADAPTER_REF=<hf-user>/<adapter-repo-or-local-path>
export FINETREE_QWEN_QUANTIZATION=bnb_8bit
export FINETREE_POD_DEBUG_ERRORS=0
```

`FINETREE_ADAPTER_REF` overrides `inference.adapter_path` for local pod inference and is the recommended
way to point the pod to a fine-tuned LoRA adapter repo.

Set `FINETREE_POD_DEBUG_ERRORS=1` during deployment/debug sessions to include exception details in `500` responses.
Each internal failure returns an `error_id` and logs a matching traceback, which you can locate with:

```bash
scripts/runpod_machine_tools.sh logs-find <error_id>
```

Build and push a Pod image:

```bash
export IMAGE_NAME=<registry>/<namespace>/finetree-pod
export IMAGE_TAG=latest
docker buildx build --platform linux/amd64 -f deploy/runpod/Dockerfile.pod -t "${IMAGE_NAME}:${IMAGE_TAG}" --push .
```

Recommended post-deploy warmup flow (first request can take time for HF download + weight materialization):

```bash
git clone https://github.com/DelmedigoA/FineTree.git
cd FineTree
export POD_ID=<your-pod-id>
export FINETREE_POD_API_KEY=<your-pod-api-key>
./scripts/runpod_api_warmup.sh --pod-id "${POD_ID}" --max-tokens 10
```

The warmup script retries until the pod can return a real `/v1/chat/completions` response.

Export quantized artifacts (default 8-bit):

```bash
finetree-ft-export-quantized --config configs/finetune_qwen35a3_vl.yaml
```

Supported request `input` keys:

- `image_path` (local path inside container) OR `image_data_uri` OR `image_base64`
- `image_mime_type` (used with `image_base64`, default `image/png`)
- `prompt` (optional; default uses FineTree extraction prompt)
- `response_mode`: `page_extraction` (default) or `text`
- `model` (optional inference model override)
- `config_path` (optional fine-tune YAML path override)

Minimal API request body example:

```json
{
  "input": {
    "image_base64": "<base64-png-or-jpg>",
    "image_mime_type": "image/png",
    "response_mode": "page_extraction"
  }
}
```

### Controls

- Draw new bbox: click + drag on image
- Jump to a page: use top-bar **Go to page** selector
- Fast browse pages: scroll and click the left **Pages** thumbnail strip
- Default page zoom: each page opens fit-to-panel-height
- Magnifier: toggle **Lens** in the top bar for zoomed cursor inspection
- Multi-select bboxes: **Shift + drag** on page
- Batch path/reference edit for selected bboxes: use **Batch Edit Selected BBoxes** (Add Parent/Child, Insert At Position, Remove First/Last, Clear Refference)
- Drag bbox: click and move selected box
- Resize bbox: drag an edge/corner of the selected box
- Edit fact: double-click inside bbox (or select bbox and click **Edit Selected Fact**)
- Duplicate bbox: **Ctrl+D**
- Gemini ground-truth draft for current page: **Gemini GT** button (**Ctrl+G**) with live streaming popup
- Qwen ground-truth draft for current page: **Qwen GT** button (**Ctrl+Shift+G**) with live streaming popup
- Delete bbox: **Delete**
- Save annotations: **Ctrl+S**
- Undo / Redo: **Ctrl+Z** / **Ctrl+Y** (also **Ctrl+Shift+Z**)
- Zoom in/out: **Ctrl + Mouse Wheel**, **Ctrl+=**, **Ctrl+-**
- Fit to page: **Ctrl+0**
- Move selected bbox(es): **Arrow keys** (hold **Shift** for faster move)
- Pan view: **Ctrl+Arrow keys** or right/middle mouse drag

### Metadata behavior

- `entity_name` is automatically copied from the current page to the next page when navigating/saving page state.

### Fact field lists

- `currency` options: `ILS`, `USD`, `EUR`, `GBP`
- `scale` options: `1`, `1000`, `1000000`

### Tests

```bash
pytest -q
```

### Build + Push Docker Image

RunPod GPU image (Qwen3.5 MoE stack, `linux/amd64`):

```bash
export IMAGE_NAME=<registry>/<namespace>/finetree-runpod
export IMAGE_TAG=qwen35-moe
docker buildx build --platform linux/amd64 -f Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}" --push .
```

Local CPU image (macOS/PC data tooling, no Unsloth GPU training):

```bash
export LOCAL_IMAGE_NAME=<registry>/<namespace>/finetree-local
export LOCAL_IMAGE_TAG=cpu
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.mac -t "${LOCAL_IMAGE_NAME}:${LOCAL_IMAGE_TAG}" --push .
```

Note: `Dockerfile.mac` is for CLI/data tooling and does not include `PyQt5`.
