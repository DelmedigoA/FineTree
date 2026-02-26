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

### Controls

- Draw new bbox: click + drag on image
- Drag bbox: click and move selected box
- Resize bbox: drag an edge/corner of the selected box
- Edit fact: double-click inside bbox (or select bbox and click **Edit Selected Fact**)
- Duplicate bbox: **Ctrl+D**
- Delete bbox: **Delete**
- Save annotations: **Ctrl+S**
- Undo / Redo: **Ctrl+Z** / **Ctrl+Y** (also **Ctrl+Shift+Z**)
- Zoom in/out: **Ctrl + Mouse Wheel**, **Ctrl+=**, **Ctrl+-**
- Fit to page: **Ctrl+0**
- Pan view: **Arrow keys** (hold **Shift** for faster pan)

### Metadata behavior

- `entity_name` is automatically copied from the current page to the next page when navigating/saving page state.

### Fact field lists

- `currency` options: `ILS`, `USD`, `EUR`, `GBP`
- `scale` options: `1`, `1000`, `1000000`

### Tests

```bash
pytest -q
```
