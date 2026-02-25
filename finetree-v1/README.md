# FineTree v1

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install poppler
```

## Convert PDF to images

Put PDFs in `data/raw/`.

```bash
python scripts/pdf_to_images.py data/raw/example.pdf
```

This creates images like `data/pages/<doc_id>/page_0001.png`.

## Label in Label Studio

Open Label Studio manually, create a project, import tasks from `data/pages/`, annotate, export JSON to `annotation/export/`.

## Parse Label Studio export

```bash
python scripts/parse_ls_export.py annotation/export/example_export.json
```

This writes structured output to `data/processed/example_export.json`.
