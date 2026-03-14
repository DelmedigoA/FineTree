# Benchmark

This workspace is for evaluating already-generated prediction JSON files against ground-truth JSON documents.

The benchmark module does not run model inference. You can use it in two ways:

- manual mode: place prediction files in `benchmark/input`, open the local UI, fill metadata, and upload `logging.jsonl`
- `info.json` mode: point `benchmark.input_dir` at a submission folder containing `info.json`, `logging.jsonl`, and `predictions/`; the UI will ingest the metadata automatically and skip the manual form

## Workspace Layout

```text
benchmark/
  config.example.yaml
  input/
    info.json
    logging.jsonl
    predictions/
      pred_0001.json
  output/
    submissions/
      <israel-local-timestamp>__<checkpoint-slug>/
        logging.jsonl
        report.json
```

## Prepare A Submission

1. Create a submission folder with:
   - `info.json`
   - `logging.jsonl`
   - prediction JSONs such as `predictions/pred_0001.json`
2. Update [`benchmark/config.example.yaml`](config.example.yaml):
   - benchmark input/output paths
   - weighting and evaluation settings
   - optional manual UI metadata defaults
   - explicit prediction-to-GT mappings
3. In `info.json`, keep `model_metadata` aligned with the benchmark submission fields and include the structured sections:
   - `training_args`
   - `environment`
   - `run`
   - `selected_checkpoint`
   - `artifacts`
4. Make the YAML mappings match the prediction filenames the submission folder contains, for example `predictions/pred_0001.json`.

## Run The Benchmark

```bash
finetree-benchmark --config benchmark/config.example.yaml --host 127.0.0.1 --port 8123
```

Open the submission page at [http://127.0.0.1:8123/submission](http://127.0.0.1:8123/submission), review the mapping status, upload `logging.jsonl`, and submit one benchmark run.

If `benchmark.input_dir` already contains `info.json`, the page will use those files directly and the manual form will not be shown.

For headless evaluation on the benchmark machine after unpacking a Colab-generated bundle:

```bash
finetree-benchmark-run \
  --config benchmark/config.example.yaml \
  --submission /path/to/run/benchmark_submission
```

## Leaderboard

Open [http://127.0.0.1:8123/leaderboard](http://127.0.0.1:8123/leaderboard).

The leaderboard supports:

- sorting by any column
- global text filtering across metrics, metadata, and logging summaries
- comparing persisted submissions without rerunning evaluation

## Reported Metrics

The persisted `report.json` and leaderboard include:

- `overall_score`
- `meta_score`
- `facts_score`
- `meta_hard_score`
- `entity_score`
- `title_score`
- `page_num_score`
- `page_type_score`
- `statement_type_score`
- logging-derived summaries such as latest/best loss, token accuracy, memory, speed, and step progress
- file-backed submission context such as run id, selected checkpoint, and artifact paths when loaded from `info.json`

## Notebook Flow

Use one of these Colab notebooks, depending on whether you are backfilling an existing model or running a new training job:

- [notebooks/qwen35_finetree_v21_infer_only.ipynb](../notebooks/qwen35_finetree_v21_infer_only.ipynb)
  Purpose: one-time backfill for the already-trained merged model `asafd60/Qwen3.5-27B-FineTree-V2.1`
- [notebooks/ms_swift_qwen35_train_and_bundle.ipynb](../notebooks/ms_swift_qwen35_train_and_bundle.ipynb)
  Purpose: future MS-Swift training runs that should emit a benchmark-ready bundle automatically

The inference-only notebook:

- downloads the exact historical `args.json` from the Hugging Face model repo
- fetches or uploads the historical `logging.jsonl`
- runs dataset-wide GPU inference for the configured target model
- writes `info.json` plus benchmark-ready predictions
- packages a self-contained submission bundle for download

The training notebook:

- saves args and environment metadata before training
- picks the single best checkpoint by lowest `eval_loss`
- pushes adapter-only exports for the best and final checkpoints
- runs inference once for the selected best checkpoint
- writes `info.json` plus benchmark-ready predictions
- packages a self-contained submission bundle for download

After downloading that bundle, move it to the benchmark/UI machine, unpack it, and either:

- point `benchmark.input_dir` at the unpacked folder and use the UI
- run `finetree-benchmark-run` against the unpacked folder
