from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field


REPORT_SCHEMA_VERSION = 1


class LoggingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    row_count: int
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


class SubmissionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_schema_version: int = REPORT_SCHEMA_VERSION
    submission_id: str
    submission_dir: str
    report_timestamp_israel: str
    config_path: str
    config_snapshot: dict[str, Any]
    submission_metadata: dict[str, Any]
    submission_source: str = "manual"
    submission_context: dict[str, Any] = Field(default_factory=dict)
    logging: LoggingReport
    aggregate_metrics: dict[str, Any]
    mapping_results: list[dict[str, Any]]
    leaderboard_row: dict[str, Any]


def israel_now(timezone_name: str = "Asia/Jerusalem") -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip().lower())
    normalized = normalized.strip("-._")
    return normalized or "submission"


def build_submission_id(timestamp: datetime, checkpoint_name: str) -> str:
    return f"{timestamp.strftime('%Y%m%dT%H%M%S%z')}__{slugify(checkpoint_name)}"


def persist_submission_report(
    *,
    output_dir: Path,
    submission_id: str,
    report: SubmissionReport,
    logging_text: str,
) -> Path:
    submission_dir = output_dir / "submissions" / submission_id
    submission_dir.mkdir(parents=True, exist_ok=False)
    (submission_dir / "logging.jsonl").write_text(logging_text, encoding="utf-8")
    (submission_dir / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return submission_dir


def _scalarize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _flatten_context(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, nested_value in value.items():
            key_name = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(_flatten_context(key_name, nested_value))
        return flattened
    return {prefix: _scalarize(value)}


def build_leaderboard_row(
    *,
    submission_id: str,
    report_timestamp_israel: str,
    submission_metadata: dict[str, Any],
    aggregate_metrics: dict[str, Any],
    logging_summary: dict[str, Any],
    submission_source: str = "manual",
    submission_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "submission_id": submission_id,
        "report_timestamp_israel": report_timestamp_israel,
        "submission_source": submission_source,
    }
    for key, value in submission_metadata.items():
        row[key] = _scalarize(value)
    for key, value in aggregate_metrics.items():
        if key == "field_scores":
            continue
        row[key] = _scalarize(value)
    for key, value in logging_summary.items():
        if key in {"latest_train_row", "latest_eval_row", "available_keys"}:
            continue
        row[key] = _scalarize(value)
    for key, value in (submission_context or {}).items():
        if key == "artifacts":
            row.update(_flatten_context("artifact", value))
            continue
        row.update(_flatten_context(key, value))
    return row


def load_reports(output_dir: Path) -> list[SubmissionReport]:
    submissions_dir = output_dir / "submissions"
    if not submissions_dir.is_dir():
        return []
    reports: list[SubmissionReport] = []
    for report_path in sorted(submissions_dir.glob("*/report.json"), reverse=True):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            reports.append(SubmissionReport.model_validate(payload))
        except Exception:
            continue
    return reports


LEADERBOARD_METRIC_COLUMNS = [
    "overall_score",
    "meta_score",
    "facts_score",
    "meta_hard_score",
    "entity_score",
    "title_score",
    "page_num_score",
    "page_type_score",
    "statement_type_score",
    "facts_count_score",
    "date_score",
    "row_role_score",
    "comment_ref_score",
    "note_flag_score",
    "note_name_score",
    "note_num_score",
    "note_ref_score",
    "path_source_score",
    "currency_score",
    "scale_score",
    "value_type_score",
    "value_context_score",
]


def leaderboard_columns(rows: list[dict[str, Any]]) -> list[str]:
    ordered = ["model", "report_timestamp_israel", "dataset", *LEADERBOARD_METRIC_COLUMNS]
    return [key for key in ordered if any(key in row for row in rows)]


def leaderboard_details(report: SubmissionReport) -> dict[str, Any]:
    return {
        "submission_id": report.submission_id,
        "submission_source": report.submission_source,
        "report_timestamp_israel": report.report_timestamp_israel,
        "submission_metadata": report.submission_metadata,
        "submission_context": report.submission_context,
        "aggregate_metrics": report.aggregate_metrics,
        "logging_summary": report.logging.summary,
    }
