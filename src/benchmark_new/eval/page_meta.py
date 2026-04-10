from __future__ import annotations

import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from ..evaluation_specs import get_evaluator_spec
from ..io.ground_truth import load_ground_truth_page
from ..models import PageMetaFieldSummary, PageMetaSummary, to_jsonable
from .scoring import evaluate_page_meta


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(pstdev(values))


def _weighted_mean(values: list[tuple[float, float]], *, normalize_weights: bool) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    if normalize_weights:
        return float(sum(score * (weight / total_weight) for score, weight in values))
    return float(sum(score * weight for score, weight in values))


def summarize_page_meta_bundle(
    bundle: dict[str, Any],
    *,
    run_dir: Path,
    data_root: Path,
) -> PageMetaSummary:
    evaluator = get_evaluator_spec("page_meta")
    field_scores: dict[str, list[float]] = {
        field.field: []
        for section in evaluator.sections
        for field in section.fields
    }
    page_count = 0
    documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
    for doc_id in sorted(documents.keys()):
        prediction_document = documents.get(doc_id)
        if not isinstance(prediction_document, dict):
            continue
        for page_entry in prediction_document.get("pages", []):
            if not isinstance(page_entry, dict):
                continue
            page_index = int(page_entry.get("page_index") or 0)
            if page_index < 1:
                continue
            gt_page = load_ground_truth_page(doc_id, page_index, data_root=data_root)
            pred_page = page_entry.get("parsed_page") if isinstance(page_entry.get("parsed_page"), dict) else {}
            gt_meta = gt_page.get("meta") if isinstance(gt_page.get("meta"), dict) else {}
            pred_meta = pred_page.get("meta") if isinstance(pred_page.get("meta"), dict) else {}
            meta_result, _ = evaluate_page_meta(gt_meta, pred_meta)
            for field_name, result in meta_result.items():
                field_scores.setdefault(field_name, []).append(float(result.score))
            page_count += 1

    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    field_summaries: list[PageMetaFieldSummary] = []
    for section in evaluator.sections:
        for field in section.fields:
            scores = field_scores.get(field.field, [])
            include_std = bool(evaluator.report.include_std) if evaluator.report is not None else False
            field_summaries.append(
                PageMetaFieldSummary(
                    field=field.field,
                    label=field.label,
                    key=field.key,
                    aggregation=field.aggregation,
                    summary_metric=field.summary_metric,
                    metrics=field.metrics,
                    weight=field.weight,
                    score=_mean(scores),
                    evaluated_page_count=len(scores),
                    std=_std(scores) if include_std else None,
                )
            )
    global_aggregation = evaluator.global_aggregation
    overall_score = _weighted_mean(
        [(field.score, field.weight) for field in field_summaries],
        normalize_weights=bool(global_aggregation.normalize_weights) if global_aggregation is not None else True,
    )
    return PageMetaSummary(
        benchmark_version=evaluator.benchmark_version,
        evaluator=evaluator.name,
        label=evaluator.label,
        run_dir=Path(run_dir),
        dataset_version_id=None if manifest.get("dataset_version_id") in (None, "") else str(manifest.get("dataset_version_id")),
        dataset_name=None if manifest.get("dataset_name") in (None, "") else str(manifest.get("dataset_name")),
        split=None if manifest.get("split") in (None, "") else str(manifest.get("split")),
        global_aggregation_method=None if global_aggregation is None else global_aggregation.method,
        overall_score=overall_score,
        page_count=page_count,
        fields=tuple(field_summaries),
    )


def write_page_meta_summary(path: Path, summary: PageMetaSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_page_meta_summary(summary: PageMetaSummary) -> str:
    headers = ("field", "aggregation", "score", "pages")
    rows = [
        (field.label, field.aggregation, f"{field.score:.6f}", str(field.evaluated_page_count))
        for field in summary.fields
    ]
    widths = [
        max([len(headers[index])] + [len(row[index]) for row in rows])
        for index in range(len(headers))
    ]
    lines = [
        summary.label,
        f"  Overall Score: {summary.overall_score:.6f}",
        f"  Pages Evaluated: {summary.page_count}",
        "",
        "  Metrics",
        f"  {' '.join(header.ljust(widths[index]) for index, header in enumerate(headers))}",
    ]
    lines.append(f"  {' '.join('-' * width for width in widths)}")
    lines.extend(
        f"  {' '.join(value.ljust(widths[index]) for index, value in enumerate(row))}"
        for row in rows
    )
    return "\n".join(lines)
