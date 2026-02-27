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

### Controls

- Draw new bbox: click + drag on image
- Multi-select bboxes: **Shift + drag** on page
- Batch path edit for selected bboxes: use **Batch Edit Selected BBoxes** (Add Parent/Child, Remove First/Last)
- Drag bbox: click and move selected box
- Resize bbox: drag an edge/corner of the selected box
- Edit fact: double-click inside bbox (or select bbox and click **Edit Selected Fact**)
- Duplicate bbox: **Ctrl+D**
- Gemini ground-truth draft for current page: **Gemini GT** button (**Ctrl+G**) with live streaming popup
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
