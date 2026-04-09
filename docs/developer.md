# Developer Guide

## Scope

This document covers day-to-day development in the current repository: local setup, test execution, common workflows, and contribution standards.

## Local Setup

Prerequisites:

- Python 3.12+
- Poppler installed for `pdf2image`
- a GUI session for the Qt dashboard and annotator

Recommended setup:

```bash
python3 -m venv .env
./.env/bin/python -m pip install --upgrade pip
./.env/bin/python -m pip install -e .
```

Optional training extras on Linux:

```bash
./.env/bin/python -m pip install -e ".[train]"
```

## Running The App

Launch the dashboard:

```bash
./.env/bin/python -m finetree_annotator
```

Launch against a PDF:

```bash
./.env/bin/python -m finetree_annotator /absolute/path/to/report.pdf
```

Launch against an extracted image directory:

```bash
./.env/bin/python -m finetree_annotator /absolute/path/to/page_images
```

## Running Tests

Fast validation:

```bash
./.env/bin/pytest -q tests/test_app_startup.py tests/test_workspace.py
```

GUI shell and annotator tests:

```bash
./.env/bin/pytest -q tests/test_dashboard_shell.py tests/test_app_layout.py
```

Model/provider flows:

```bash
./.env/bin/pytest -q tests/test_gemini_vlm.py tests/test_qwen_vlm.py tests/test_pod_api.py tests/test_pod_gradio.py
```

Benchmark:

```bash
./.env/bin/pytest -q tests/test_benchmark_new.py
```

Full suite:

```bash
./.env/bin/pytest -q
```

## Adding Features

Preferred sequence:

1. Put canonical data rules in a non-Qt module first.
2. Add or update tests for normalization, schema handling, or evaluation behavior before wiring UI.
3. Keep Qt widgets thin and move non-visual logic into reusable helpers.
4. Route long-running work through worker objects on `QThread`; do not block the UI thread with PDF conversion, model calls, or dataset export.
5. Update docs when adding a new CLI, config field, or workflow.

## Coding Standards

Repository expectations:

- Keep business logic outside the Qt layer whenever possible.
- Use explicit type hints on public helpers and new modules.
- Prefer deterministic transforms over implicit mutation.
- Normalize IO at module boundaries.
- Use canonical schema models from [schemas.py](/Users/delmedigo/Dev/FineTree/src/finetree_annotator/schemas.py) and [schema_io.py](/Users/delmedigo/Dev/FineTree/src/finetree_annotator/schema_io.py).
- Keep logging structured and file-backed for model calls and benchmark/evaluation runs.
- Preserve backward compatibility only where existing data files require it.

What to avoid:

- new logic in `build/`, `dist/`, notebook outputs, or log directories
- Qt-side network calls or PDF rendering on the main thread
- duplicated schema normalization or duplicated startup/import logic
- new tracked generated artifacts under `gemini_logs/`, `qwen_logs/`, `benchmark_new/outputs/`, or backup directories

## Config Conventions

- Fine-tuning config lives under `configs/` and is validated by [finetune/config.py](/Users/delmedigo/Dev/FineTree/src/finetree_annotator/finetune/config.py).
- Benchmark configuration and scoring live under [src/benchmark_new](/Users/delmedigo/Dev/FineTree/src/benchmark_new).
- Prompt templates live under `prompts/`.
- Do not hardcode environment-specific secrets or endpoints; use env vars or YAML config.

## Remote Development

Typical split:

- local machine: annotation, quick tests, dataset inspection, benchmark evaluation
- remote machine or RunPod: fine-tuning, model serving, and batch inference generation

Relevant CLIs:

- `finetree-ft-build-dataset`
- `finetree-ft-train`
- `finetree-ft-merge-push`
- `finetree-ft-export-quantized`
- `finetree-pod-api`
- `finetree-pod-gradio`
- `finetree-runpod-worker`

## Repository Hygiene

Treat these as generated outputs:

- `gemini_logs/`
- `qwen_logs/`
- `artifacts/`
- `benchmark_new/outputs/`
- `build/`
- `dist/`
- `data/annotations/_backup/`
- `data/annotations_backup_*`

If a test needs fixture data, keep the fixture small, explicit, and isolated under a stable test-owned path rather than relying on large tracked workspace state.
