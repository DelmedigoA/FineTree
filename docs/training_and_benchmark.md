# Training And Benchmark Guide

## Scope

This document covers the model-facing half of FineTree: dataset generation, training configuration, remote execution, prediction packaging, and benchmark evaluation.

## Dataset Build

The dataset pipeline starts from canonical annotation JSON under `data/annotations/`.

Primary config:

- [configs/finetune_qwen35a3_vl.yaml](/Users/delmedigo/Dev/FineTree/configs/finetune_qwen35a3_vl.yaml)
- validator and schema: [finetune/config.py](/Users/delmedigo/Dev/FineTree/src/finetree_annotator/finetune/config.py)

Build train and validation JSONL:

```bash
./.env/bin/finetree-ft-build-dataset --config configs/finetune_qwen35a3_vl.yaml
```

Expected outputs:

- `data/finetune/train.jsonl`
- `data/finetune/val.jsonl`

Important dataset controls:

- `data.val_ratio`
- `data.val_doc_ids`
- `data.include_empty_pages`
- `data.bbox_policy`
- `data.fact_order_enforce`
- `prompt.prompt_path`

## Hugging Face Dataset Export

BBox-preserving push flow:

```bash
./.env/bin/finetree-ft-push-dataset --config configs/finetune_qwen35a3_vl.yaml
```

BBox-free push flow:

```bash
./.env/bin/finetree-ft-push-dataset-no-bbox --config configs/finetune_qwen35a3_vl.yaml
```

Use the bbox-free flow when the consumer model or training setup should not see geometric labels in the target JSON.

## Training

Validate config and environment first:

```bash
./.env/bin/finetree-ft-preflight --config configs/finetune_qwen35a3_vl.yaml
```

Run training:

```bash
./.env/bin/finetree-ft-train --config configs/finetune_qwen35a3_vl.yaml
```

Merge and push trained artifacts:

```bash
./.env/bin/finetree-ft-merge-push --config configs/finetune_qwen35a3_vl.yaml
```

Export quantized artifacts:

```bash
./.env/bin/finetree-ft-export-quantized --config configs/finetune_qwen35a3_vl.yaml
```

## Remote Execution

The repository includes RunPod-focused assets under:

- [deploy/runpod](/Users/delmedigo/Dev/FineTree/deploy/runpod)
- [scripts/runpod_bootstrap.sh](/Users/delmedigo/Dev/FineTree/scripts/runpod_bootstrap.sh)
- [scripts/runpod_train.sh](/Users/delmedigo/Dev/FineTree/scripts/runpod_train.sh)
- [scripts/runpod_smoke_test.sh](/Users/delmedigo/Dev/FineTree/scripts/runpod_smoke_test.sh)

Typical remote split:

1. build or sync datasets locally
2. upload configs and scripts to the remote machine
3. run preflight and training remotely
4. run prediction/inference remotely or against a served endpoint
5. bring benchmark bundle back to the benchmark machine

## Prediction Structure

The benchmark accepts page-level prediction files or canonical document JSON depending on config.

Example page-level prediction:

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

The exact schema is defined in [PAGE_LEVEL_PREDICTED_SCHEMA.md](/Users/delmedigo/Dev/FineTree/PAGE_LEVEL_PREDICTED_SCHEMA.md).

## Benchmark Inputs

Benchmark config:

- [benchmark/config.example.yaml](/Users/delmedigo/Dev/FineTree/benchmark/config.example.yaml)

Expected submission bundle shape:

```text
benchmark_submission/
  info.json
  logging.jsonl
  predictions/
    pred_0001.json
    pred_0002.json
    ...
```

`info.json` should carry model metadata and any run context that you want persisted into the benchmark report.

## Benchmark UI

Start the local web UI:

```bash
./.env/bin/finetree-benchmark --config benchmark/config.example.yaml --host 127.0.0.1 --port 8123
```

Submission page:

- `/submission`

Leaderboard page:

- `/leaderboard`

The UI persists reports under `benchmark/output/submissions/<submission_id>/`.

## Headless Benchmark Run

Run a bundle without opening the UI:

```bash
./.env/bin/finetree-benchmark-run \
  --config benchmark/config.example.yaml \
  --submission /absolute/path/to/benchmark_submission
```

## Reproducing Results

To reproduce a historical benchmark result:

1. check out the commit that produced the model-serving and scoring behavior
2. use the same benchmark config and mappings
3. use the same prediction bundle or rerun inference against the same dataset version
4. keep `benchmark.timezone` at `Asia/Jerusalem` so submission IDs and report timestamps remain comparable

## Benchmark Outputs

Each persisted report includes:

- submission metadata
- logging summary
- mapping-level metrics
- aggregate metrics
- leaderboard row data

The headless runner prints a small JSON summary to stdout with `submission_id`, `submission_dir`, and `overall_score`.
