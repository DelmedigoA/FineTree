from __future__ import annotations

import curses
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from .eval import (
    evaluate_predictions_bundle,
    format_facts_summary,
    format_page_meta_summary,
    summarize_facts_run,
    summarize_page_meta_bundle,
    write_facts_summary,
    write_page_meta_summary,
)
from .evaluation_specs import get_evaluator_spec
from .io import list_dataset_versions, load_predictions_from_run_dir
from .io.workspace_paths import DEFAULT_DATA_ROOT
from .models import DatasetVersionInfo, NativeRunInfo
from .reports import (
    build_path_comparison_report,
    write_mistakes_values_json,
    write_path_comparison_report,
    write_path_mistakes_json,
    write_values_mistakes_json,
)


T = TypeVar("T")
DEFAULT_OUTPUT_ROOT = Path("benchmark_new/outputs")
EVALUATION_TARGETS: tuple[str, ...] = ("meta", "facts", "both")


def dataset_option_label(dataset: DatasetVersionInfo) -> str:
    val_stats = dict(dataset.split_stats.get("val") or {})
    return (
        f"{dataset.name} | {dataset.version_id} | "
        f"val_docs={val_stats.get('doc_count', 0)} | val_pages={val_stats.get('page_count', 0)}"
    )


def run_option_label(run: NativeRunInfo) -> str:
    return f"{run.model_name} ({run.run_id})"


def evaluation_target_label(target: str) -> str:
    labels = {
        "meta": "meta",
        "facts": "facts",
        "both": "both",
    }
    return labels[target]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_native_inference_runs(
    dataset_version_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[NativeRunInfo]:
    runs: list[NativeRunInfo] = []
    root = Path(output_root)
    if not root.is_dir():
        return runs
    for manifest_path in sorted(root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        if not (run_dir / "documents").is_dir():
            continue
        manifest = _load_json(manifest_path)
        if str(manifest.get("dataset_version_id") or "") != dataset_version_id:
            continue
        if str(manifest.get("split") or "") != "val":
            continue
        runs.append(
            NativeRunInfo(
                run_id=run_dir.name,
                run_dir=run_dir,
                model_name=str(manifest.get("model") or "unknown-model"),
                dataset_version_id=str(manifest.get("dataset_version_id") or "") or None,
                dataset_name=str(manifest.get("dataset_name") or "") or None,
                split=str(manifest.get("split") or "") or None,
                started_at=float(manifest["started_at"]) if manifest.get("started_at") is not None else None,
            )
        )
    runs.sort(key=lambda item: (item.started_at is not None, item.started_at or float("-inf"), item.run_id), reverse=True)
    return runs


def _render_menu(
    stdscr: curses.window,
    *,
    title: str,
    options: Sequence[T],
    render_label: Callable[[T], str],
) -> T | None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.keypad(True)
    selected_index = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        header = [
            title,
            "Use Up/Down arrows and Enter to confirm. Press q or Esc to exit.",
            "",
        ]
        for index, line in enumerate(header):
            stdscr.addnstr(index, 0, line, max(width - 1, 1))
        visible_rows = max(height - len(header), 1)
        start = 0
        if selected_index >= visible_rows:
            start = selected_index - visible_rows + 1
        for row_offset, option in enumerate(options[start : start + visible_rows]):
            option_index = start + row_offset
            label = render_label(option)
            prefix = "> " if option_index == selected_index else "  "
            stdscr.addnstr(
                len(header) + row_offset,
                0,
                prefix + label,
                max(width - 1, 1),
                curses.A_REVERSE if option_index == selected_index else curses.A_NORMAL,
            )
        stdscr.refresh()
        key = stdscr.getch()
        if key in {ord("q"), 27}:
            return None
        if key in {curses.KEY_UP, ord("k")}:
            selected_index = (selected_index - 1) % len(options)
            continue
        if key in {curses.KEY_DOWN, ord("j")}:
            selected_index = (selected_index + 1) % len(options)
            continue
        if key in {curses.KEY_ENTER, 10, 13}:
            return options[selected_index]


def pick_option_curses(
    title: str,
    options: Sequence[T],
    render_label: Callable[[T], str],
) -> T | None:
    if not options:
        return None
    return curses.wrapper(lambda stdscr: _render_menu(stdscr, title=title, options=options, render_label=render_label))


def execute_interactive_evaluation(
    evaluation_target: str,
    *,
    run: NativeRunInfo,
    data_root: Path = DEFAULT_DATA_ROOT,
    stdout: object | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    bundle = load_predictions_from_run_dir(run.run_dir, data_root=Path(data_root))
    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    page_meta_summary = None
    facts_summary = None
    if evaluation_target in {"meta", "both"}:
        page_meta_summary = summarize_page_meta_bundle(bundle, run_dir=run.run_dir, data_root=Path(data_root))
        page_meta_spec = get_evaluator_spec("page_meta")
        output_file = page_meta_spec.output_file or "page_meta_summary.json"
        artifact_path = run.run_dir / "evaluation" / output_file
        write_page_meta_summary(artifact_path, page_meta_summary)
        print(format_page_meta_summary(page_meta_summary), file=output)
        print("", file=output)
        print("  Artifacts", file=output)
        print(f"  Page Meta Summary: {artifact_path}", file=output)
    if evaluation_target in {"facts", "both"}:
        run_result = evaluate_predictions_bundle(bundle, data_root=Path(data_root))
        facts_summary = summarize_facts_run(run_result, run_dir=run.run_dir, manifest=manifest)
        artifact_path = run.run_dir / "evaluation" / "facts_summary.json"
        write_facts_summary(artifact_path, facts_summary)
        values_report_path = run.run_dir / "evaluation" / "mistakes_values.json"
        write_mistakes_values_json(
            values_report_path,
            run_result=run_result,
            bundle=bundle,
            data_root=Path(data_root),
        )
        simple_values_report_path = run.run_dir / "evaluation" / "values_mistakes.json"
        write_values_mistakes_json(
            simple_values_report_path,
            run_result=run_result,
            bundle=bundle,
            data_root=Path(data_root),
        )
        simple_path_report_path = run.run_dir / "evaluation" / "path_mistakes.json"
        write_path_mistakes_json(
            simple_path_report_path,
            run_result=run_result,
            bundle=bundle,
            data_root=Path(data_root),
        )
        comparison_report = build_path_comparison_report(run_result=run_result, bundle=bundle, data_root=Path(data_root))
        comparison_report_path = run.run_dir / "evaluation" / "path_comparison_report.json"
        write_path_comparison_report(comparison_report_path, comparison_report)
        print(format_facts_summary(facts_summary), file=output)
        print("", file=output)
        print("  Artifacts", file=output)
        print(f"  Facts Summary: {artifact_path}", file=output)
        print(f"  Detailed Values Report: {values_report_path}", file=output)
        print(f"  Simple Values Mistakes: {simple_values_report_path}", file=output)
        print(f"  Simple Path Mistakes: {simple_path_report_path}", file=output)
        print(f"  Path Comparison Report: {comparison_report_path}", file=output)
    if page_meta_summary is not None and facts_summary is not None:
        final_score = float((page_meta_summary.overall_score + facts_summary.overall_score) / 2.0)
        print("", file=output)
        print("Final Score", file=output)
        print(f"  Page Meta: {page_meta_summary.overall_score:.6f}", file=output)
        print(f"  Facts: {facts_summary.overall_score:.6f}", file=output)
        print(f"  Combined: {final_score:.6f}", file=output)
    return 0


def run_interactive_benchmark(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    picker: Callable[[str, Sequence[T], Callable[[T], str]], T | None] | None = None,
    stdout: object | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    option_picker = picker or pick_option_curses
    datasets = list_dataset_versions(data_root=Path(data_root))
    if not datasets:
        print(f"No datasets found under {Path(data_root) / 'datasets'}.", file=output)
        return 1
    dataset = option_picker("Choose dataset to evaluate", datasets, dataset_option_label)
    if dataset is None:
        return 0
    runs = list_native_inference_runs(dataset.version_id, output_root=Path(output_root))
    if not runs:
        print(
            f"No compatible native benchmark runs found for dataset={dataset.version_id} split=val.",
            file=output,
        )
        return 1
    run = option_picker("Choose previous inference", runs, run_option_label)
    if run is None:
        return 0
    evaluation_target = option_picker("Choose what to evaluate", EVALUATION_TARGETS, evaluation_target_label)
    if evaluation_target is None:
        return 0
    return execute_interactive_evaluation(str(evaluation_target), run=run, data_root=Path(data_root), stdout=output)
