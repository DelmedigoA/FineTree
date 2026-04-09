# Training And Evaluation Guide

## Scope

This document covers the model-facing half of FineTree: dataset generation, training configuration, remote execution, prediction generation, and benchmark evaluation.

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
5. run `benchmark-new` locally against the selected dataset version and outputs

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

The exact schema is defined in [PAGE_LEVEL_PREDICTED_SCHEMA.md](/Users/delmedigo/Dev/FineTree/docs/schemas/PAGE_LEVEL_PREDICTED_SCHEMA.md).

## Benchmark Inputs

Canonical benchmark selection comes from saved dataset versions in the web GUI. `benchmark-new` resolves the chosen version and split directly from workspace data.

Typical generated run layout:

```text
benchmark_new/outputs/<timestamp>_<provider>_<dataset>/
  manifest.json
  progress.json
  events.jsonl
  requests.jsonl
  documents/<doc_id>/pages/page_XXXX.json
  evaluation/run_metrics.json
  evaluation/summary.csv
  evaluation/full_metrics.csv
```

## Benchmark CLI

List available dataset versions:

```bash
./.env/bin/benchmark-new datasets list
```

Run provider inference only:

```bash
./.env/bin/benchmark-new infer \
  --dataset-version-id <dataset_version_id> \
  --provider gemini \
  --model <model_name>
```

Run evaluation on an existing output directory:

```bash
./.env/bin/benchmark-new eval --run-dir /absolute/path/to/run_dir
```

Run inference and evaluation end to end:

```bash
./.env/bin/benchmark-new batch-eval \
  --dataset-version-id <dataset_version_id> \
  --provider finetree_vllm \
  --model <model_name> \
  --base-url <openai_compatible_endpoint>
```

## Reproducing Results

To reproduce a historical benchmark result:

1. check out the commit that produced the model-serving and scoring behavior
2. use the same dataset version id and split
3. use the same provider settings or rerun inference against the same endpoint
4. compare `summary.csv` and `run_metrics.json` from the persisted run directory

## Benchmark Outputs

Each persisted report includes:

- run metadata
- provider and request logs
- page/document/run metrics
- `summary.csv`
- `full_metrics.csv`
- `run_metrics.json`
