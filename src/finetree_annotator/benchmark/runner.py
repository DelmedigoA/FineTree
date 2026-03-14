from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, load_benchmark_config
from .info import load_submission_info_bundle
from .inputs import prepare_mapping
from .logging_summary import summarize_logging_rows
from .scoring import aggregate_mapping_metrics, evaluate_document_detailed
from .storage import (
    LoggingReport,
    SubmissionReport,
    build_leaderboard_row,
    build_submission_id,
    israel_now,
    persist_submission_report,
)


@dataclass(frozen=True)
class SubmissionPayload:
    submission_metadata: dict[str, Any]
    logging_filename: str
    logging_text: str
    logging_rows: list[dict[str, Any]]
    source: str = "manual"
    source_submission_dir: Path | None = None
    submission_context: dict[str, Any] = field(default_factory=dict)


def _evaluate_mappings(cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    mapping_results: list[dict[str, Any]] = []
    for mapping in cfg.mappings:
        prepared = prepare_mapping(cfg, mapping)
        metrics = evaluate_document_detailed(
            prepared.prediction_document,
            prepared.ground_truth_document,
            meta_rules=cfg.weighting.meta_fields,
            aggregate_weights=cfg.weighting.aggregate,
            facts_method=cfg.methods.facts,
        )
        mapping_results.append(
            {
                "prediction_file": str(mapping.prediction_file),
                "prediction_path": str(prepared.prediction_path),
                "gt_file": str(mapping.gt_file),
                "gt_path": str(prepared.gt_path),
                "prediction_format": prepared.prediction_format,
                "gt_page_index": mapping.gt_page_index,
                "metrics": metrics,
            }
        )
    return mapping_results


def _mirror_report_to_source(source_submission_dir: Path, report: SubmissionReport) -> Path:
    report_path = source_submission_dir / "report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def run_submission(
    *,
    cfg: BenchmarkConfig,
    cfg_path: Path,
    submission: SubmissionPayload,
    input_dir_override: Path | None = None,
) -> tuple[SubmissionReport, Path]:
    effective_cfg = cfg.model_copy(deep=True)
    if input_dir_override is not None:
        effective_cfg.benchmark.input_dir = input_dir_override.expanduser().resolve()

    logging_summary = summarize_logging_rows(submission.logging_rows)
    mapping_results = _evaluate_mappings(effective_cfg)
    aggregate_metrics = aggregate_mapping_metrics(mapping_results)
    report_timestamp = israel_now(effective_cfg.benchmark.timezone)
    checkpoint_name = str(submission.submission_metadata["checkpoint_name"])
    submission_id = build_submission_id(report_timestamp, checkpoint_name)
    leaderboard_row = build_leaderboard_row(
        submission_id=submission_id,
        report_timestamp_israel=report_timestamp.isoformat(),
        submission_metadata=submission.submission_metadata,
        aggregate_metrics=aggregate_metrics,
        logging_summary=logging_summary,
        submission_source=submission.source,
        submission_context=submission.submission_context,
    )
    submission_dir_str = str(effective_cfg.benchmark.output_dir / "submissions" / submission_id)
    report = SubmissionReport(
        submission_id=submission_id,
        submission_dir=submission_dir_str,
        report_timestamp_israel=report_timestamp.isoformat(),
        config_path=str(cfg_path),
        config_snapshot=effective_cfg.model_dump(mode="json"),
        submission_metadata=submission.submission_metadata,
        submission_source=submission.source,
        submission_context=submission.submission_context,
        logging=LoggingReport(
            filename=submission.logging_filename,
            row_count=len(submission.logging_rows),
            rows=submission.logging_rows,
            summary=logging_summary,
        ),
        aggregate_metrics=aggregate_metrics,
        mapping_results=mapping_results,
        leaderboard_row=leaderboard_row,
    )
    submission_dir = persist_submission_report(
        output_dir=effective_cfg.benchmark.output_dir,
        submission_id=submission_id,
        report=report,
        logging_text=submission.logging_text,
    )
    if submission.source_submission_dir is not None:
        _mirror_report_to_source(submission.source_submission_dir, report)
    return report, submission_dir


def run_info_submission(*, config_path: Path | str, submission_dir: Path | str) -> tuple[SubmissionReport, Path]:
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = load_benchmark_config(cfg_path)
    cfg.benchmark.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_submission_info_bundle(submission_dir)
    payload = SubmissionPayload(
        submission_metadata=bundle.info.model_metadata,
        logging_filename=bundle.logging_path.name,
        logging_text=bundle.logging_text,
        logging_rows=bundle.logging_rows,
        source="info_json",
        source_submission_dir=bundle.submission_dir,
        submission_context=bundle.context,
    )
    return run_submission(
        cfg=cfg,
        cfg_path=cfg_path,
        submission=payload,
        input_dir_override=bundle.submission_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FineTree benchmark submission without starting the web UI.")
    parser.add_argument("--config", required=True, help="Benchmark YAML config path.")
    parser.add_argument(
        "--submission",
        required=True,
        help="Submission folder containing info.json, logging.jsonl, and mapped prediction files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, submission_dir = run_info_submission(config_path=args.config, submission_dir=args.submission)
    print(
        json.dumps(
            {
                "status": "ok",
                "submission_id": report.submission_id,
                "submission_dir": str(submission_dir),
                "overall_score": report.aggregate_metrics.get("overall_score"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
