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

Recommended workflow on a fresh RunPod pod:

```bash
cd /workspace
git clone <YOUR_REPO_URL> FineTree
cd FineTree
./scripts/runpod_bootstrap.sh
```

If you use this as pod start command, use:

```bash
bash -lc "cd /workspace/FineTree && ./scripts/runpod_bootstrap.sh && sleep infinity"
```

`sleep infinity` prevents RunPod from repeatedly restarting the container command and re-running install steps.

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

Optional manual steps:

```bash
finetree-ft-build-dataset --config configs/finetune_qwen35a3_vl.yaml
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml --dry-run
finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml
```

Enable Hub push in config only when needed:

```bash
# push_to_hub.enabled: true
# push_to_hub.repo_id: <your_org>/<your_model_repo>
# keep push_to_hub.hf_token: null (use FINETREE_HF_TOKEN env var)
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
