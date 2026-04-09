# FineTree

FineTree is a desktop-first workflow for annotating financial-report pages, building training datasets, serving multimodal extraction models, and evaluating model runs against ground truth.

The repository contains four connected surfaces:

- a PyQt5 desktop dashboard and annotator for page-level review
- dataset-building and Hugging Face export tooling for training runs
- local and RunPod-facing inference and serving entrypoints
- a standalone benchmark runner for comparing model outputs against ground truth

## Key Features

- Open a PDF or an extracted image directory and annotate one page at a time.
- Store annotations as canonical JSON with schema normalization and equation integrity checks.
- Generate or patch annotations with Gemini and Qwen-backed flows.
- Build page-level training JSONL files and push dataset exports to Hugging Face.
- Serve models through FastAPI, Gradio, RunPod queue, or RunPod serverless entrypoints.
- Score model outputs against ground truth with persisted run reports and CSV metrics.

## Example Output

Canonical page-level output is a JSON object with page metadata plus ordered facts:

```json
{
  "meta": {
    "entity_name": "צלול - עמותה לאיכות הסביבה (ע״ר)",
    "page_num": null,
    "page_type": "title",
    "statement_type": null,
    "title": "דוח שנתי 2024"
  },
  "facts": []
}
```

The exact predicted page schema is documented in [PAGE_LEVEL_PREDICTED_SCHEMA.md](/Users/delmedigo/Dev/FineTree/docs/schemas/PAGE_LEVEL_PREDICTED_SCHEMA.md).

## Quickstart

Prerequisites:

- Python 3.12+
- Poppler utilities available on `PATH` for `pdf2image`
- A desktop session if you want to launch the Qt annotator

Copy-paste local setup:

```bash
python3 -m venv .env
./.env/bin/python -m pip install --upgrade pip
./.env/bin/python -m pip install -e .
./.env/bin/pytest -q tests/test_app_startup.py tests/test_workspace.py
./.env/bin/python -m finetree_annotator
```

Open a PDF directly:

```bash
./.env/bin/python -m finetree_annotator /absolute/path/to/report.pdf
```

Run a benchmark dataset listing:

```bash
./.env/bin/benchmark-new datasets list
```

## Main Entry Points

Installed console scripts:

- `finetree-annotator`
- `finetree-ft-build-dataset`
- `finetree-ft-push-dataset`
- `finetree-ft-push-dataset-no-bbox`
- `finetree-ft-train`
- `finetree-ft-merge-push`
- `finetree-ft-export-quantized`
- `benchmark-new`
- `benchmark_new.cli`
- `finetree-pod-api`
- `finetree-pod-gradio`
- `finetree-simple-infer-api`
- `finetree-runpod-worker`
- `finetree-runpod-pod-start`

List them locally with:

```bash
./.env/bin/python -m finetree_annotator.list_commands
```

## Project Structure

Current layout:

```text
src/finetree_annotator/
  app.py                  Qt annotator window
  dashboard.py            Qt dashboard shell and workspace browser
  workspace.py            managed workspace IO and PDF import helpers
  gemini_vlm.py           Gemini generation, streaming, parsing, and logging
  qwen_vlm.py             Qwen local/endpoint inference flows
  ai/                     UI-facing AI dialog, controller, bbox helpers
src/benchmark_new/        standalone benchmark package, CLI, inference, evaluation
  deploy/                 FastAPI, Gradio, RunPod serverless/pod entrypoints
  finetune/               dataset build, validation, train, export, push flows
  schemas.py              canonical Pydantic models
  schema_*.py             schema registry, guards, contracts, IO helpers

configs/                  local and remote config files
scripts/                  one-off utilities and repo maintenance scripts
tests/                    pytest suite
benchmark_new/            benchmark run inputs and outputs kept out of source package
docs/                     developer, architecture, and workflow docs
data/                     managed PDFs, extracted images, annotations, backups
prompts/                  extraction and Gemini prompt templates
```

Generated and experimental content also exists in this repository today, especially under `gemini_logs/`, `qwen_logs/`, `build/`, `dist/`, `artifacts/`, and backup-heavy `data/` paths. Those should be treated as outputs, not as primary source code.

## Documentation

- Developer guide: [docs/developer.md](/Users/delmedigo/Dev/FineTree/docs/developer.md)
- Architecture guide: [docs/architecture.md](/Users/delmedigo/Dev/FineTree/docs/architecture.md)
- Training and evaluation guide: [docs/training_and_benchmark.md](/Users/delmedigo/Dev/FineTree/docs/training_and_benchmark.md)
- Workflow overview: [docs/finetree_workflow.md](/Users/delmedigo/Dev/FineTree/docs/finetree_workflow.md)
- Benchmark package README: [src/benchmark_new/README.md](/Users/delmedigo/Dev/FineTree/src/benchmark_new/README.md)

## Typical Workflow

1. Import a PDF into the dashboard and annotate pages in the desktop app.
2. Save canonical page JSON under `data/annotations/`.
3. Build training JSONL from approved annotations.
4. Train or export models locally or on a remote machine.
5. Run predictions through Gemini or the FineTree vLLM endpoint.
6. Evaluate outputs through `benchmark-new infer`, `benchmark-new eval`, or `benchmark-new batch-eval`.

## Testing

Run the full suite:

```bash
./.env/bin/pytest -q
```

Run focused suites while working on a subsystem:

```bash
./.env/bin/pytest -q tests/test_dashboard_shell.py tests/test_app_layout.py
./.env/bin/pytest -q tests/test_gemini_vlm.py tests/test_qwen_vlm.py
./.env/bin/pytest -q tests/test_benchmark_new.py
```

## Backup

Create a workspace backup archive under `backups/`:

```bash
PYTHONPATH=src python -m finetree_annotator.workspace_backup
```

By default this snapshots `data/raw_pdfs`, `data/pdf_images`, `data/annotations`, `data/finetune`, `data/workspace_review_state.json`, and `db/finetree.db`, while excluding generated noise such as `data/doctr_logs/` and annotation backup history.

## Current Caveats

- The annotator and dashboard code still live in large modules and need further decomposition.
- Generated logs and fixture-like data are still tracked in Git history and inflate the repository.
- `build/`, `dist/`, and log directories reflect generated outputs and should not be treated as authoritative source.
