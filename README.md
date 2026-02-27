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

Validate environment/config without training:

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --dry-run
```

Optional merge after training:

```bash
finetree-ft-merge-push --config configs/finetune_qwen35a3_vl.yaml
```

### RunPod GPU Validation + Full Fine-Tune

On a fresh RunPod GPU VM:

```bash
git clone <YOUR_REPO_URL> FineTree
cd FineTree
python3 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

GPU preflight tests (CUDA + stack + dataset smoke):

```bash
export FINETREE_RUN_GPU_TESTS=1
pytest -q -m gpu
```

Fail-fast preflight command (recommended before any training run):

```bash
finetree-ft-preflight --config configs/finetune_qwen35a3_vl.yaml
```

Include Hugging Face connectivity/auth validation:

```bash
finetree-ft-preflight --config configs/finetune_qwen35a3_vl.yaml --check-hf
```

Run GPU tests plus explicit HF API test:

```bash
export FINETREE_RUN_GPU_TESTS=1
export FINETREE_RUN_HF_TESTS=1
pytest -q -m "gpu or hf"
```

Optional full test suite:

```bash
pytest -q
```

Enable Hub push in config and set your repo id:

```bash
# edit configs/finetune_qwen35a3_vl.yaml:
# push_to_hub.enabled: true
# push_to_hub.repo_id: <your_org>/<your_model_repo>
# keep push_to_hub.hf_token: null (do not commit secrets)
```

Set HF token via environment (recommended):

```bash
export FINETREE_HF_TOKEN=<your_hf_token>
```

Build datasets:

```bash
finetree-ft-build-dataset --config configs/finetune_qwen35a3_vl.yaml
```

Training dry-run (verifies CUDA + deps + dataset pathing):

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --dry-run
```

Run full fine-tune:

```bash
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml
```

Optional merge after training:

```bash
finetree-ft-merge-push --config configs/finetune_qwen35a3_vl.yaml
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
