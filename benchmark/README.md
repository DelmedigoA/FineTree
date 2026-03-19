# Benchmark

The benchmark evaluates already-generated prediction JSON files against ground-truth annotation documents. It does not run model inference itself.

Use it when you already have:

- `predictions/*.json`
- `logging.jsonl`
- optional `info.json` metadata bundle

## Modes

### Web UI

Start the local benchmark server:

```bash
./.env/bin/finetree-benchmark --config benchmark/config.example.yaml --host 127.0.0.1 --port 8123
```

Open:

- [http://127.0.0.1:8123/submission](http://127.0.0.1:8123/submission)
- [http://127.0.0.1:8123/leaderboard](http://127.0.0.1:8123/leaderboard)

### Headless Runner

Run a benchmark bundle directly:

```bash
./.env/bin/finetree-benchmark-run \
  --config benchmark/config.example.yaml \
  --submission /absolute/path/to/benchmark_submission
```

## Expected Layout

```text
benchmark_submission/
  info.json
  logging.jsonl
  predictions/
    pred_0001.json
```

The benchmark config controls the explicit mappings between prediction files and ground-truth pages.

## Config

Start from [config.example.yaml](/Users/delmedigo/Dev/FineTree/benchmark/config.example.yaml).

Important sections:

- `benchmark.input_dir`
- `benchmark.output_dir`
- `methods`
- `weighting`
- `evaluation`
- `mappings`

## Outputs

Persisted reports are written under:

```text
benchmark/output/submissions/<submission_id>/
  logging.jsonl
  report.json
```

## More Detail

For the end-to-end dataset, training, prediction, and evaluation flow, see [docs/training_and_benchmark.md](/Users/delmedigo/Dev/FineTree/docs/training_and_benchmark.md).
