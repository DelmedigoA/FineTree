from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..algorithms import align_sequences_by_index, sequence_ratio
from ..io.ground_truth import load_ground_truth_page
from ..models import RunResult, to_jsonable
from ..spec import FACT_FIELDS, META_FIELDS, DATE_FACT_FIELDS, SOFT_STRING_FACT_FIELDS, normalize_fact_value, normalize_meta_value, parse_date_value, parse_numeric_value


def _mean_float(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summary_metric_fieldnames() -> list[str]:
    fieldnames: list[str] = []
    for spec in META_FIELDS:
        prefix = f"meta__{spec.field}"
        fieldnames.extend(
            [
                f"{prefix}__score",
                f"{prefix}__precision",
                f"{prefix}__recall",
                f"{prefix}__f1",
                f"{prefix}__accuracy",
            ]
        )
    for spec in FACT_FIELDS:
        prefix = f"facts__{spec.field}"
        fieldnames.extend(
            [
                f"{prefix}__score",
                f"{prefix}__precision",
                f"{prefix}__recall",
                f"{prefix}__f1",
                f"{prefix}__accuracy",
                f"{prefix}__string_similarity",
                f"{prefix}__numeric_mae",
                f"{prefix}__date_mae_days",
            ]
        )
    return fieldnames


def _build_summary_metric_values(pages: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in META_FIELDS:
        prefix = f"meta__{spec.field}"
        field_results = [page.meta_result[spec.field] for page in pages if spec.field in page.meta_result]
        if not field_results:
            continue
        out[f"{prefix}__score"] = f"{_mean_float([item.score for item in field_results]):.6f}"
        exact_values = [1.0 if item.is_exact_match else 0.0 for item in field_results]
        out[f"{prefix}__precision"] = f"{_mean_float(exact_values):.6f}"
        out[f"{prefix}__recall"] = f"{_mean_float(exact_values):.6f}"
        out[f"{prefix}__f1"] = f"{_mean_float(exact_values):.6f}"
        out[f"{prefix}__accuracy"] = f"{_mean_float(exact_values):.6f}"

    for spec in FACT_FIELDS:
        prefix = f"facts__{spec.field}"
        field_results = [page.facts_result[spec.field] for page in pages if spec.field in page.facts_result]
        if not field_results:
            continue
        out[f"{prefix}__score"] = f"{_mean_float([item.score for item in field_results]):.6f}"
        out[f"{prefix}__precision"] = f"{_mean_float([item.exact_metrics.precision for item in field_results]):.6f}"
        out[f"{prefix}__recall"] = f"{_mean_float([item.exact_metrics.recall for item in field_results]):.6f}"
        out[f"{prefix}__f1"] = f"{_mean_float([item.exact_metrics.f1 for item in field_results]):.6f}"
        out[f"{prefix}__accuracy"] = f"{_mean_float([item.exact_metrics.accuracy for item in field_results]):.6f}"
        string_values = [item.string_similarity for item in field_results if item.string_similarity is not None]
        numeric_values = [item.numeric_mae for item in field_results if item.numeric_mae is not None]
        date_values = [item.date_mae_days for item in field_results if item.date_mae_days is not None]
        out[f"{prefix}__string_similarity"] = "" if not string_values else f"{_mean_float([float(value) for value in string_values]):.6f}"
        out[f"{prefix}__numeric_mae"] = "" if not numeric_values else f"{_mean_float([float(value) for value in numeric_values]):.6f}"
        out[f"{prefix}__date_mae_days"] = "" if not date_values else f"{_mean_float([float(value) for value in date_values]):.6f}"
    return out


def write_run_metrics_json(path: Path, run_result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(run_result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _element_score(*, field: str, gt_value: Any, pred_value: Any, section: str) -> float:
    if section == "meta":
        if field in {spec.field for spec in META_FIELDS if spec.comparison_type == "soft"}:
            return sequence_ratio("" if gt_value is None else str(gt_value), "" if pred_value is None else str(pred_value))
        return 1.0 if gt_value == pred_value else 0.0
    if field == "path":
        return 1.0 if gt_value == pred_value else 0.0
    if field == "value":
        return 1.0 if gt_value == pred_value else 0.0
    if field in SOFT_STRING_FACT_FIELDS:
        return sequence_ratio("" if gt_value is None else str(gt_value), "" if pred_value is None else str(pred_value))
    return 1.0 if gt_value == pred_value else 0.0


def _serialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item) for item in value]
    return value


def write_mistakes_json(
    path: Path,
    *,
    run_result: RunResult,
    bundle: dict[str, Any],
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    documents_out: list[dict[str, Any]] = []
    bundle_documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
    meta_soft_fields = {spec.field for spec in META_FIELDS if spec.comparison_type == "soft"}

    for document in run_result.documents:
        bundle_document = bundle_documents.get(document.doc_id, {}) if isinstance(bundle_documents.get(document.doc_id), dict) else {}
        page_lookup = {
            int(page.get("page_index") or 0): page
            for page in bundle_document.get("pages", [])
            if isinstance(page, dict)
        }
        page_entries: list[dict[str, Any]] = []
        for page in document.pages:
            gt_page = load_ground_truth_page(document.doc_id, page.page_index, data_root=data_root)
            pred_entry = page_lookup.get(page.page_index, {})
            pred_page = pred_entry.get("parsed_page") if isinstance(pred_entry.get("parsed_page"), dict) else {}

            meta_mistakes: list[dict[str, Any]] = []
            gt_meta = gt_page.get("meta") if isinstance(gt_page.get("meta"), dict) else {}
            pred_meta = pred_page.get("meta") if isinstance(pred_page.get("meta"), dict) else {}
            for spec in META_FIELDS:
                gt_value = normalize_meta_value(spec.field, gt_meta.get(spec.field))
                pred_value = normalize_meta_value(spec.field, pred_meta.get(spec.field))
                score = _element_score(field=spec.field, gt_value=gt_value, pred_value=pred_value, section="meta")
                if score >= 1.0:
                    continue
                meta_mistakes.append(
                    {
                        "field": spec.field,
                        "comparison_type": spec.comparison_type,
                        "score": score,
                        "gt_value": _serialize_value(gt_value),
                        "pred_value": _serialize_value(pred_value),
                    }
                )

            fact_mistakes: list[dict[str, Any]] = []
            gt_facts = gt_page.get("facts") if isinstance(gt_page.get("facts"), list) else []
            pred_facts = pred_page.get("facts") if isinstance(pred_page.get("facts"), list) else []
            for spec in FACT_FIELDS:
                gt_sequence = [normalize_fact_value(spec.field, fact.get(spec.field)) for fact in gt_facts if isinstance(fact, dict)]
                pred_sequence = [normalize_fact_value(spec.field, fact.get(spec.field)) for fact in pred_facts if isinstance(fact, dict)]
                mismatches: list[dict[str, Any]] = []
                for index, (gt_value, pred_value) in enumerate(align_sequences_by_index(gt_sequence, pred_sequence), start=1):
                    score = _element_score(field=spec.field, gt_value=gt_value, pred_value=pred_value, section="facts")
                    if score >= 1.0:
                        continue
                    item: dict[str, Any] = {
                        "element_index": index,
                        "score": score,
                        "gt_value": _serialize_value(gt_value),
                        "pred_value": _serialize_value(pred_value),
                    }
                    if spec.field == "value":
                        item["gt_numeric"] = parse_numeric_value(gt_value)
                        item["pred_numeric"] = parse_numeric_value(pred_value)
                    if spec.field in DATE_FACT_FIELDS:
                        item["gt_date"] = None if parse_date_value(gt_value) is None else str(parse_date_value(gt_value))
                        item["pred_date"] = None if parse_date_value(pred_value) is None else str(parse_date_value(pred_value))
                    mismatches.append(item)
                if mismatches:
                    fact_result = page.facts_result[spec.field]
                    fact_mistakes.append(
                        {
                            "field": spec.field,
                            "field_score": fact_result.score,
                            "exact_metrics": to_jsonable(fact_result.exact_metrics),
                            "string_similarity": fact_result.string_similarity,
                            "numeric_mae": fact_result.numeric_mae,
                            "date_mae_days": fact_result.date_mae_days,
                            "elements": mismatches,
                        }
                    )

            if meta_mistakes or fact_mistakes:
                page_entries.append(
                    {
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "meta_score": page.meta_score,
                        "facts_score": page.facts_score,
                        "page_score": page.page_score,
                        "facts_status": page.facts_status,
                        "meta_mistakes": meta_mistakes,
                        "fact_mistakes": fact_mistakes,
                    }
                )
        if page_entries:
            documents_out.append(
                {
                    "doc_id": document.doc_id,
                    "document_score": document.document_score,
                    "pages": page_entries,
                }
            )

    payload = {
        "documents": documents_out,
        "document_count": len(documents_out),
        "page_count": sum(len(document["pages"]) for document in documents_out),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(
    path: Path,
    *,
    run_result: RunResult,
    provider: str,
    dataset_version_id: str | None,
    dataset_name: str | None,
    split: str | None,
    total_tokens_received: int,
    total_failed_pages: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_type",
                "provider",
                "dataset_version_id",
                "dataset_name",
                "split",
                "doc_id",
                "page_count",
                "gt_fact_count",
                "pred_fact_count",
                "empty_gt_pages",
                "false_positive_empty_gt_pages",
                "failed_pages",
                "tokens_received",
                "meta_score",
                "facts_score",
                "combined_score",
            ]
            + _summary_metric_fieldnames(),
        )
        writer.writeheader()
        for document in run_result.documents:
            gt_fact_count = sum(page.gt_fact_count for page in document.pages)
            pred_fact_count = sum(page.pred_fact_count for page in document.pages)
            empty_gt_pages = sum(1 for page in document.pages if page.gt_fact_count == 0)
            false_positive_empty_gt_pages = sum(1 for page in document.pages if page.facts_status == "false_positive_on_empty_gt")
            row = {
                "row_type": "document",
                "provider": provider,
                "dataset_version_id": dataset_version_id or "",
                "dataset_name": dataset_name or "",
                "split": split or "",
                "doc_id": document.doc_id,
                "page_count": len(document.pages),
                "gt_fact_count": gt_fact_count,
                "pred_fact_count": pred_fact_count,
                "empty_gt_pages": empty_gt_pages,
                "false_positive_empty_gt_pages": false_positive_empty_gt_pages,
                "failed_pages": 0,
                "tokens_received": "",
                "meta_score": f"{document.meta_score:.6f}",
                "facts_score": f"{document.facts_score:.6f}",
                "combined_score": f"{document.document_score:.6f}",
            }
            row.update(_build_summary_metric_values(document.pages))
            writer.writerow(row)
        run_pages = [page for document in run_result.documents for page in document.pages]
        row = {
            "row_type": "run",
            "provider": provider,
            "dataset_version_id": dataset_version_id or "",
            "dataset_name": dataset_name or "",
            "split": split or "",
            "doc_id": "",
            "page_count": sum(len(document.pages) for document in run_result.documents),
            "gt_fact_count": sum(page.gt_fact_count for document in run_result.documents for page in document.pages),
            "pred_fact_count": sum(page.pred_fact_count for document in run_result.documents for page in document.pages),
            "empty_gt_pages": sum(1 for document in run_result.documents for page in document.pages if page.gt_fact_count == 0),
            "false_positive_empty_gt_pages": sum(
                1 for document in run_result.documents for page in document.pages if page.facts_status == "false_positive_on_empty_gt"
            ),
            "failed_pages": total_failed_pages,
            "tokens_received": total_tokens_received,
            "meta_score": f"{run_result.meta_score:.6f}",
            "facts_score": f"{run_result.facts_score:.6f}",
            "combined_score": f"{run_result.run_score:.6f}",
        }
        row.update(_build_summary_metric_values(run_pages))
        writer.writerow(row)


def write_full_metrics_csv(
    path: Path,
    *,
    run_result: RunResult,
    provider: str,
    dataset_version_id: str | None,
    dataset_name: str | None,
    split: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "provider",
                "dataset_version_id",
                "dataset_name",
                "split",
                "doc_id",
                "page_index",
                "image_name",
                "section",
                "field",
                "score",
                "is_exact_match",
                "gt_fact_count",
                "pred_fact_count",
                "facts_status",
                "matches",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "helper_pair_count",
                "string_similarity",
                "numeric_mae",
                "date_mae_days",
                "gt_value_preview",
                "pred_value_preview",
            ],
        )
        writer.writeheader()
        for document in run_result.documents:
            for page in document.pages:
                for field_name, meta_result in page.meta_result.items():
                    writer.writerow(
                        {
                            "provider": provider,
                            "dataset_version_id": dataset_version_id or "",
                            "dataset_name": dataset_name or "",
                            "split": split or "",
                            "doc_id": document.doc_id,
                            "page_index": page.page_index,
                            "image_name": page.image_name,
                            "section": "meta",
                            "field": field_name,
                            "score": f"{meta_result.score:.6f}",
                            "is_exact_match": str(meta_result.is_exact_match).lower(),
                            "gt_fact_count": page.gt_fact_count,
                            "pred_fact_count": page.pred_fact_count,
                            "facts_status": page.facts_status,
                            "matches": "1" if meta_result.is_exact_match else "0",
                            "precision": "1.000000" if meta_result.is_exact_match else "0.000000",
                            "recall": "1.000000" if meta_result.is_exact_match else "0.000000",
                            "f1": "1.000000" if meta_result.is_exact_match else "0.000000",
                            "accuracy": "1.000000" if meta_result.is_exact_match else "0.000000",
                            "helper_pair_count": "",
                            "string_similarity": "",
                            "numeric_mae": "",
                            "date_mae_days": "",
                            "gt_value_preview": "" if meta_result.gt_value is None else str(meta_result.gt_value),
                            "pred_value_preview": "" if meta_result.pred_value is None else str(meta_result.pred_value),
                        }
                    )
                for field_name, fact_result in page.facts_result.items():
                    writer.writerow(
                        {
                            "provider": provider,
                            "dataset_version_id": dataset_version_id or "",
                            "dataset_name": dataset_name or "",
                            "split": split or "",
                            "doc_id": document.doc_id,
                            "page_index": page.page_index,
                            "image_name": page.image_name,
                            "section": "facts",
                            "field": field_name,
                            "score": f"{fact_result.score:.6f}",
                            "is_exact_match": "",
                            "gt_fact_count": page.gt_fact_count,
                            "pred_fact_count": page.pred_fact_count,
                            "facts_status": page.facts_status,
                            "matches": fact_result.exact_metrics.matches,
                            "precision": f"{fact_result.exact_metrics.precision:.6f}",
                            "recall": f"{fact_result.exact_metrics.recall:.6f}",
                            "f1": f"{fact_result.exact_metrics.f1:.6f}",
                            "accuracy": f"{fact_result.exact_metrics.accuracy:.6f}",
                            "helper_pair_count": fact_result.details.get("helper_pair_count", ""),
                            "string_similarity": "" if fact_result.string_similarity is None else f"{fact_result.string_similarity:.6f}",
                            "numeric_mae": "" if fact_result.numeric_mae is None else f"{fact_result.numeric_mae:.6f}",
                            "date_mae_days": "" if fact_result.date_mae_days is None else f"{fact_result.date_mae_days:.6f}",
                            "gt_value_preview": str(fact_result.details.get("gt_preview") or ""),
                            "pred_value_preview": str(fact_result.details.get("pred_preview") or ""),
                        }
                    )
