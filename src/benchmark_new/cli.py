from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .eval import evaluate_predictions_bundle
from .infer import create_run_dir, run_inference_pipeline
from .interactive import run_interactive_benchmark
from .io import list_dataset_versions, load_dataset_selection, load_predictions_from_run_dir
from .io.workspace_paths import DEFAULT_DATA_ROOT
from .providers import ProviderOptions
from .reports import (
    build_path_comparison_report,
    write_full_metrics_csv,
    write_path_comparison_report,
    write_mistakes_json,
    write_mistakes_values_json,
    write_path_mistakes_json,
    write_run_metrics_json,
    write_summary_csv,
    write_values_mistakes_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone FineTree benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets", help="List or inspect dataset versions.")
    datasets_subparsers = datasets.add_subparsers(dest="datasets_command", required=True)
    datasets_list = datasets_subparsers.add_parser("list", help="List dataset versions")
    datasets_list.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Workspace data root")

    infer = subparsers.add_parser("infer", help="Run provider inference on a dataset split.")
    _add_dataset_args(infer)
    _add_provider_args(infer)
    infer.add_argument("--output-root", default="benchmark_new/outputs", help="Output root directory")

    evaluate = subparsers.add_parser("eval", help="Evaluate an existing run directory.")
    evaluate.add_argument("--run-dir", required=True, help="Existing run directory")
    evaluate.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Workspace data root")

    batch_eval = subparsers.add_parser("batch-eval", help="Run inference and evaluation end to end.")
    _add_dataset_args(batch_eval)
    _add_provider_args(batch_eval)
    batch_eval.add_argument("--output-root", default="benchmark_new/outputs", help="Output root directory")
    return parser


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-version-id", required=True, help="Dataset version id from the web GUI dataset versions")
    parser.add_argument("--split", default="val", help="Dataset split to benchmark")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Workspace data root")


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", required=True, choices=["gemini", "finetree_vllm"], help="Inference provider")
    parser.add_argument("--model", required=True, help="Provider model id")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint for finetree_vllm")
    parser.add_argument("--api-key", help="Gemini API key override")
    parser.add_argument("--max-pixels", type=int, default=1_400_000)
    parser.add_argument("--max-tokens", type=int, default=24_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-level")


def _print_dataset_versions(data_root: Path) -> int:
    versions = list_dataset_versions(data_root=data_root)
    for version in versions:
        val_stats = dict(version.split_stats.get("val") or {})
        print(
            f"{version.version_id}\t{version.name}\t"
            f"val_docs={val_stats.get('doc_count', 0)}\tval_pages={val_stats.get('page_count', 0)}"
        )
    return 0


def _run_infer(args: argparse.Namespace, *, evaluate_after_infer: bool) -> int:
    data_root = Path(args.data_root)
    selection = load_dataset_selection(args.dataset_version_id, split=args.split, data_root=data_root)
    if not selection.documents:
        raise SystemExit(f"No documents found for dataset={args.dataset_version_id} split={args.split}")
    run_dir = create_run_dir(
        output_root=Path(args.output_root),
        provider=args.provider,
        dataset_label=selection.dataset.name,
    )
    options = ProviderOptions(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_pixels=args.max_pixels,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=bool(args.enable_thinking),
        thinking_level=args.thinking_level,
    )
    output = run_inference_pipeline(
        documents=selection.documents,
        options=options,
        run_dir=run_dir,
        dataset_version_id=selection.dataset.version_id,
        dataset_name=selection.dataset.name,
        split=selection.split,
        data_root=data_root,
        evaluate_after_infer=evaluate_after_infer,
    )
    print(f"run_dir={run_dir}")
    if evaluate_after_infer:
        print(f"run_score={output['run_result'].run_score:.6f}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    run_dir = Path(args.run_dir)
    bundle = load_predictions_from_run_dir(run_dir, data_root=data_root)
    run_result = evaluate_predictions_bundle(bundle, data_root=data_root)
    evaluation_dir = run_dir / "evaluation"
    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    provider = str(manifest.get("provider") or manifest.get("model") or bundle.get("kind") or "external")
    dataset_version_id = manifest.get("dataset_version_id")
    dataset_name = manifest.get("dataset_name")
    split = manifest.get("split")
    total_tokens_received = 0
    total_failed_pages = 0
    for document in bundle.get("documents", {}).values():
        for page in document.get("pages", []):
            total_tokens_received += int(page.get("received_tokens") or 0)
            if page.get("error"):
                total_failed_pages += 1
    write_run_metrics_json(evaluation_dir / "run_metrics.json", run_result)
    write_summary_csv(
        evaluation_dir / "summary.csv",
        run_result=run_result,
        provider=provider,
        dataset_version_id=dataset_version_id,
        dataset_name=dataset_name,
        split=split,
        total_tokens_received=total_tokens_received,
        total_failed_pages=total_failed_pages,
    )
    write_full_metrics_csv(
        evaluation_dir / "full_metrics.csv",
        run_result=run_result,
        provider=provider,
        dataset_version_id=dataset_version_id,
        dataset_name=dataset_name,
        split=split,
    )
    write_mistakes_json(
        evaluation_dir / "mistakes.json",
        run_result=run_result,
        bundle=bundle,
        data_root=data_root,
    )
    write_mistakes_values_json(
        evaluation_dir / "mistakes_values.json",
        run_result=run_result,
        bundle=bundle,
        data_root=data_root,
    )
    write_values_mistakes_json(
        evaluation_dir / "values_mistakes.json",
        run_result=run_result,
        bundle=bundle,
        data_root=data_root,
    )
    write_path_mistakes_json(
        evaluation_dir / "path_mistakes.json",
        run_result=run_result,
        bundle=bundle,
        data_root=data_root,
    )
    write_path_comparison_report(
        evaluation_dir / "path_comparison_report.json",
        build_path_comparison_report(run_result=run_result, bundle=bundle, data_root=data_root),
    )
    print(f"run_score={run_result.run_score:.6f}")
    print(f"evaluation_dir={evaluation_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(
                "Interactive benchmark CLI requires a TTY. Use explicit commands such as `benchmark.cli datasets list` or `benchmark.cli eval --run-dir <run_dir>`.",
                file=sys.stderr,
            )
            return 2
        return run_interactive_benchmark()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "datasets" and args.datasets_command == "list":
        return _print_dataset_versions(Path(args.data_root))
    if args.command == "infer":
        return _run_infer(args, evaluate_after_infer=False)
    if args.command == "batch-eval":
        return _run_infer(args, evaluate_after_infer=True)
    if args.command == "eval":
        return _run_eval(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
