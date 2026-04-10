# `benchmark_new`

Standalone benchmark runner for FineTree.

## Commands

```bash
benchmark.cli
python3 -m benchmark_new.cli datasets list
python3 -m benchmark_new.cli infer --dataset-version-id <id> --provider finetree_vllm --model <model> --base-url <url>
python3 -m benchmark_new.cli eval --run-dir <run_dir>
python3 -m benchmark_new.cli batch-eval --dataset-version-id <id> --provider gemini --model gemini-3-flash-preview
```

`benchmark.cli` opens the interactive benchmark wizard. Existing explicit subcommands remain available for scripted usage.

## Output layout

- `manifest.json`
- `progress.json`
- `events.jsonl`
- `requests.jsonl`
- `documents/<doc_id>/pages/page_XXXX.json`
- `documents/<doc_id>/summary.json`
- `evaluation/run_metrics.json`
- `evaluation/summary.csv`
- `evaluation/full_metrics.csv`

## Notes

- Dataset selection reads the saved web GUI dataset versions under `data/datasets`.
- The default split is `val`.
- `eval` supports both native `benchmark_new` runs and older GUI/vLLM run folders.
