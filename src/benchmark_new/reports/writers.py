from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..algorithms import align_sequences_by_index, lcs_index_pairs, sequence_ratio
from ..evaluation_specs import get_facts_field_spec
from ..io.ground_truth import load_ground_truth_page
from ..models import RunResult, to_jsonable
from ..spec import FACT_FIELDS, META_FIELDS, DATE_FACT_FIELDS, SOFT_STRING_FACT_FIELDS, normalize_fact_value, normalize_meta_value, parse_date_value, parse_numeric_value


FACT_VALUE_EVALUATOR_SPEC = get_facts_field_spec("value")
FACT_PATH_EVALUATOR_SPEC = get_facts_field_spec("path")


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
        if spec.field == "value":
            out[f"{prefix}__accuracy"] = f"{_mean_float([float(item.details.get('value_accuracy', 0.0)) for item in field_results]):.6f}"
        else:
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


def _value_report_entry(fact: dict[str, Any], index: int) -> dict[str, Any]:
    raw_value = fact.get("value")
    raw_path = fact.get("path")
    raw_fact_num = fact.get("fact_num")
    normalized_value = normalize_fact_value("value", raw_value)
    normalized_path = normalize_fact_value("path", raw_path)
    normalized_fact_num = normalize_fact_value("fact_num", raw_fact_num)
    return {
        "fact_index": index + 1,
        "fact_num": _serialize_value(normalized_fact_num),
        "path": _serialize_value(normalized_path),
        "value": _serialize_value(normalized_value),
        "raw_fact_num": _serialize_value(raw_fact_num),
        "raw_path": _serialize_value(raw_path),
        "raw_value": _serialize_value(raw_value),
        "alignment_key": _serialize_value(normalized_value),
    }


def _path_report_entry(fact: dict[str, Any], index: int) -> dict[str, Any]:
    raw_value = fact.get("value")
    raw_path = fact.get("path")
    normalized_value = normalize_fact_value("value", raw_value)
    normalized_path = normalize_fact_value("path", raw_path)
    return {
        "fact_index": index + 1,
        "value": _serialize_value(normalized_value),
        "path": _serialize_value(normalized_path),
        "raw_value": _serialize_value(raw_value),
        "raw_path": _serialize_value(raw_path),
    }


def _path_threshold_match(gt_path: Any, pred_path: Any) -> bool:
    left = list(gt_path or [])
    right = list(pred_path or [])
    threshold = 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
    require_same_length = True if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.require_same_length is None else bool(FACT_PATH_EVALUATOR_SPEC.require_same_length)
    if require_same_length and len(left) != len(right):
        return False
    if len(left) != len(right):
        return False
    return all(sequence_ratio(str(left_item), str(right_item)) >= threshold for left_item, right_item in zip(left, right))


def _path_last_leaf_threshold_match(gt_path: Any, pred_path: Any) -> bool:
    left = list(gt_path or [])
    right = list(pred_path or [])
    threshold = 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
    if not left or not right:
        return False
    return sequence_ratio(str(left[-1]), str(right[-1])) >= threshold


def _path_soft_overlap_metrics(gt_path: Any, pred_path: Any) -> dict[str, float | int]:
    left = list(gt_path or [])
    right = list(pred_path or [])
    threshold = 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
    if not left and not right:
        return {"shared_nodes": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    rows = len(left)
    cols = len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for left_index in range(rows):
        for right_index in range(cols):
            if sequence_ratio(str(left[left_index]), str(right[right_index])) >= threshold:
                table[left_index + 1][right_index + 1] = table[left_index][right_index] + 1
            else:
                table[left_index + 1][right_index + 1] = max(table[left_index][right_index + 1], table[left_index + 1][right_index])
    shared_nodes = table[rows][cols]
    precision = float(shared_nodes / len(right)) if right else (1.0 if not left else 0.0)
    recall = float(shared_nodes / len(left)) if left else (1.0 if not right else 0.0)
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    return {"shared_nodes": int(shared_nodes), "precision": precision, "recall": recall, "f1": f1}


def _is_subsequence(left: list[str], right: list[str]) -> bool:
    if not left:
        return True
    cursor = 0
    for item in right:
        if cursor < len(left) and left[cursor] == item:
            cursor += 1
    return cursor == len(left)


def _classify_path_error(gt_path: list[str], pred_path: list[str], *, strict_result: bool, leaf_result: bool) -> str:
    gt_nodes = list(gt_path or [])
    pred_nodes = list(pred_path or [])
    if not leaf_result:
        return "wrong_leaf"
    gt_ancestors = gt_nodes[:-1]
    pred_ancestors = pred_nodes[:-1]
    if len(gt_ancestors) > len(pred_ancestors) and _is_subsequence(pred_ancestors, gt_ancestors):
        return "missing_ancestor"
    if len(pred_ancestors) > len(gt_ancestors) and _is_subsequence(gt_ancestors, pred_ancestors):
        return "extra_ancestor"
    return "wrong_ancestor"


def _sort_report_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            str(item.get("doc_id") or ""),
            int(item.get("page_index") or 0),
            str(item.get("image_name") or ""),
            str(item.get("value") or ""),
            json.dumps(item.get("gt_path") or [], ensure_ascii=False),
            json.dumps(item.get("pred_path") or [], ensure_ascii=False),
        ),
    )


def _dedupe_report_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in _sort_report_items(items):
        fingerprint = json.dumps(
            {
                "doc_id": item.get("doc_id"),
                "page_index": item.get("page_index"),
                "image_name": item.get("image_name"),
                "gt_path": item.get("gt_path"),
                "pred_path": item.get("pred_path"),
                "strict": item.get("strict"),
                "leaf": item.get("leaf"),
                "soft_overlap": item.get("soft_overlap"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(item)
    return deduped


def build_path_comparison_report(
    *,
    run_result: RunResult,
    bundle: dict[str, Any],
    data_root: Path,
    max_examples: int = 8,
) -> dict[str, Any]:
    bundle_documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
    grouped_items: dict[str, list[dict[str, Any]]] = {
        "missing_ancestor": [],
        "extra_ancestor": [],
        "wrong_ancestor": [],
        "wrong_leaf": [],
    }

    for document in run_result.documents:
        bundle_document = bundle_documents.get(document.doc_id, {}) if isinstance(bundle_documents.get(document.doc_id), dict) else {}
        page_lookup = {
            int(page.get("page_index") or 0): page
            for page in bundle_document.get("pages", [])
            if isinstance(page, dict)
        }
        for page in document.pages:
            gt_page = load_ground_truth_page(document.doc_id, page.page_index, data_root=data_root)
            pred_entry = page_lookup.get(page.page_index, {})
            pred_page = pred_entry.get("parsed_page") if isinstance(pred_entry.get("parsed_page"), dict) else {}
            gt_facts = [fact for fact in gt_page.get("facts", []) if isinstance(fact, dict)]
            pred_facts = [fact for fact in pred_page.get("facts", []) if isinstance(fact, dict)]
            gt_values = [_value_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_values = [_value_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            gt_paths = [_path_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_paths = [_path_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            matched_pairs = lcs_index_pairs(
                [entry["alignment_key"] for entry in gt_values],
                [entry["alignment_key"] for entry in pred_values],
            )
            for gt_index, pred_index in matched_pairs:
                gt_item = gt_paths[gt_index]
                pred_item = pred_paths[pred_index]
                strict_result = _path_threshold_match(gt_item["path"], pred_item["path"])
                leaf_result = _path_last_leaf_threshold_match(gt_item["path"], pred_item["path"])
                soft_metrics = _path_soft_overlap_metrics(gt_item["path"], pred_item["path"])
                if strict_result or (not leaf_result and float(soft_metrics["f1"]) < 0.75):
                    continue
                error_type = _classify_path_error(list(gt_item["path"] or []), list(pred_item["path"] or []), strict_result=strict_result, leaf_result=leaf_result)
                grouped_items[error_type].append(
                    {
                        "doc_id": document.doc_id,
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "value": gt_item["value"],
                        "gt_path": gt_item["path"],
                        "pred_path": pred_item["path"],
                        "strict": int(strict_result),
                        "leaf": int(leaf_result),
                        "soft_overlap": {
                            "precision": float(soft_metrics["precision"]),
                            "recall": float(soft_metrics["recall"]),
                            "f1": float(soft_metrics["f1"]),
                        },
                    }
                )

    counts = {error_type: len(items) for error_type, items in grouped_items.items()}
    ordered_types = sorted(grouped_items.keys(), key=lambda error_type: (-counts[error_type], error_type))
    selected_examples: list[dict[str, Any]] = []
    grouped_examples: dict[str, list[dict[str, Any]]] = {}
    remaining = max_examples
    for error_type in ordered_types:
        if remaining <= 0:
            grouped_examples[error_type] = []
            continue
        sorted_items = _dedupe_report_items(grouped_items[error_type])
        take = min(2, len(sorted_items), remaining)
        chosen = sorted_items[:take]
        grouped_examples[error_type] = chosen
        selected_examples.extend(chosen)
        remaining -= take

    if len(selected_examples) < 5:
        leftovers: list[dict[str, Any]] = []
        for error_type in ordered_types:
            shown = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in grouped_examples.get(error_type, [])}
            for item in _dedupe_report_items(grouped_items[error_type]):
                fingerprint = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if fingerprint not in shown:
                    leftovers.append(item)
        for item in leftovers[: max(0, 5 - len(selected_examples))]:
            selected_examples.append(item)
            grouped_examples.setdefault(_classify_path_error(item["gt_path"], item["pred_path"], strict_result=bool(item["strict"]), leaf_result=bool(item["leaf"])), []).append(item)

    examples_by_type = [
        {"error_type": error_type, "count": counts[error_type], "examples": grouped_examples.get(error_type, [])}
        for error_type in ordered_types
        if counts[error_type] > 0
    ]
    representative_examples = _dedupe_report_items(selected_examples)[:3]
    return {
        "matcher": "lcs",
        "matching_on": list(FACT_VALUE_EVALUATOR_SPEC.alignment_on) if FACT_VALUE_EVALUATOR_SPEC is not None else ["value"],
        "threshold": 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold),
        "example_count": sum(len(group["examples"]) for group in examples_by_type),
        "counts": counts,
        "groups": examples_by_type,
        "representative_examples": representative_examples,
    }


def write_path_comparison_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_path_comparison_report_summary(report: dict[str, Any]) -> str:
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    lines = [
        "Path Comparison Report",
        "  Error Type Counts",
    ]
    for error_type, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
        if int(count) <= 0:
            continue
        lines.append(f"  {error_type}: {int(count)}")
    examples = report.get("representative_examples") if isinstance(report.get("representative_examples"), list) else []
    if examples:
        lines.append("")
        lines.append("  Representative Examples")
        for item in examples[:3]:
            soft_overlap = item.get("soft_overlap", {}) if isinstance(item.get("soft_overlap"), dict) else {}
            lines.append(
                "  "
                + f"{item.get('doc_id')} page={item.get('page_index')} "
                + f"gt={item.get('gt_path')} pred={item.get('pred_path')} "
                + f"strict={item.get('strict')} leaf={item.get('leaf')} "
                + f"soft_f1={float(soft_overlap.get('f1', 0.0)):.3f}"
            )
    return "\n".join(lines)


def write_mistakes_values_json(
    path: Path,
    *,
    run_result: RunResult,
    bundle: dict[str, Any],
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    documents_out: list[dict[str, Any]] = []
    bundle_documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}

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
            gt_facts = [fact for fact in gt_page.get("facts", []) if isinstance(fact, dict)]
            pred_facts = [fact for fact in pred_page.get("facts", []) if isinstance(fact, dict)]

            gt_values = [_value_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_values = [_value_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            gt_alignment_keys = [entry["alignment_key"] for entry in gt_values]
            pred_alignment_keys = [entry["alignment_key"] for entry in pred_values]
            matched_pairs = lcs_index_pairs(gt_alignment_keys, pred_alignment_keys)
            matched_gt_indices = {gt_index for gt_index, _ in matched_pairs}
            matched_pred_indices = {pred_index for _, pred_index in matched_pairs}

            matched_pairs_out: list[dict[str, Any]] = []
            mismatched_pairs_out: list[dict[str, Any]] = []
            for pair_index, (gt_index, pred_index) in enumerate(matched_pairs, start=1):
                gt_item = gt_values[gt_index]
                pred_item = pred_values[pred_index]
                is_value_match = gt_item["value"] == pred_item["value"]
                pair_payload = {
                    "pair_index": pair_index,
                    "gt_fact_index": gt_item["fact_index"],
                    "pred_fact_index": pred_item["fact_index"],
                    "alignment_key": gt_item["alignment_key"],
                    "value_match": is_value_match,
                    "gt": gt_item,
                    "pred": pred_item,
                }
                matched_pairs_out.append(pair_payload)
                if not is_value_match:
                    mismatched_pairs_out.append(pair_payload)

            unmatched_gt = [gt_values[index] for index in range(len(gt_values)) if index not in matched_gt_indices]
            unmatched_pred = [pred_values[index] for index in range(len(pred_values)) if index not in matched_pred_indices]

            page_entries.append(
                {
                    "page_index": page.page_index,
                    "image_name": page.image_name,
                    "facts_status": page.facts_status,
                    "gt_fact_count": page.gt_fact_count,
                    "pred_fact_count": page.pred_fact_count,
                    "gt_values": gt_values,
                    "pred_values": pred_values,
                    "value_matching": {
                        "matcher": "lcs",
                        "alignment_on": list(FACT_VALUE_EVALUATOR_SPEC.alignment_on) if FACT_VALUE_EVALUATOR_SPEC is not None else ["value"],
                        "matched_pair_count": len(matched_pairs_out),
                        "mismatched_pair_count": len(mismatched_pairs_out),
                        "unmatched_gt_count": len(unmatched_gt),
                        "unmatched_pred_count": len(unmatched_pred),
                        "matched_pairs": matched_pairs_out,
                        "mismatched_pairs": mismatched_pairs_out,
                        "unmatched_gt": unmatched_gt,
                        "unmatched_pred": unmatched_pred,
                    },
                }
            )

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


def write_values_mistakes_json(
    path: Path,
    *,
    run_result: RunResult,
    bundle: dict[str, Any],
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
    items: list[dict[str, Any]] = []

    for document in run_result.documents:
        bundle_document = bundle_documents.get(document.doc_id, {}) if isinstance(bundle_documents.get(document.doc_id), dict) else {}
        page_lookup = {
            int(page.get("page_index") or 0): page
            for page in bundle_document.get("pages", [])
            if isinstance(page, dict)
        }
        for page in document.pages:
            gt_page = load_ground_truth_page(document.doc_id, page.page_index, data_root=data_root)
            pred_entry = page_lookup.get(page.page_index, {})
            pred_page = pred_entry.get("parsed_page") if isinstance(pred_entry.get("parsed_page"), dict) else {}
            gt_facts = [fact for fact in gt_page.get("facts", []) if isinstance(fact, dict)]
            pred_facts = [fact for fact in pred_page.get("facts", []) if isinstance(fact, dict)]

            gt_values = [_value_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_values = [_value_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            matched_pairs = lcs_index_pairs(
                [entry["alignment_key"] for entry in gt_values],
                [entry["alignment_key"] for entry in pred_values],
            )
            matched_gt_indices = {gt_index for gt_index, _ in matched_pairs}
            matched_pred_indices = {pred_index for _, pred_index in matched_pairs}

            for gt_index, pred_index in matched_pairs:
                gt_item = gt_values[gt_index]
                pred_item = pred_values[pred_index]
                if gt_item["value"] == pred_item["value"]:
                    continue
                items.append(
                    {
                        "doc_id": document.doc_id,
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "kind": "matched_pair_mismatch",
                        "gt": gt_item,
                        "pred": pred_item,
                    }
                )

            for gt_index, gt_item in enumerate(gt_values):
                if gt_index in matched_gt_indices:
                    continue
                items.append(
                    {
                        "doc_id": document.doc_id,
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "kind": "unmatched_gt",
                        "gt": gt_item,
                        "pred": None,
                    }
                )

            for pred_index, pred_item in enumerate(pred_values):
                if pred_index in matched_pred_indices:
                    continue
                items.append(
                    {
                        "doc_id": document.doc_id,
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "kind": "unmatched_pred",
                        "gt": None,
                        "pred": pred_item,
                    }
                )

    payload = {
        "mistake_count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_path_mistakes_json(
    path: Path,
    *,
    run_result: RunResult,
    bundle: dict[str, Any],
    data_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_documents = bundle.get("documents", {}) if isinstance(bundle.get("documents"), dict) else {}
    items: list[dict[str, Any]] = []

    for document in run_result.documents:
        bundle_document = bundle_documents.get(document.doc_id, {}) if isinstance(bundle_documents.get(document.doc_id), dict) else {}
        page_lookup = {
            int(page.get("page_index") or 0): page
            for page in bundle_document.get("pages", [])
            if isinstance(page, dict)
        }
        for page in document.pages:
            gt_page = load_ground_truth_page(document.doc_id, page.page_index, data_root=data_root)
            pred_entry = page_lookup.get(page.page_index, {})
            pred_page = pred_entry.get("parsed_page") if isinstance(pred_entry.get("parsed_page"), dict) else {}
            gt_facts = [fact for fact in gt_page.get("facts", []) if isinstance(fact, dict)]
            pred_facts = [fact for fact in pred_page.get("facts", []) if isinstance(fact, dict)]

            gt_values = [_value_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_values = [_value_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            gt_paths = [_path_report_entry(fact, index) for index, fact in enumerate(gt_facts)]
            pred_paths = [_path_report_entry(fact, index) for index, fact in enumerate(pred_facts)]
            matched_pairs = lcs_index_pairs(
                [entry["alignment_key"] for entry in gt_values],
                [entry["alignment_key"] for entry in pred_values],
            )

            for gt_index, pred_index in matched_pairs:
                gt_item = gt_paths[gt_index]
                pred_item = pred_paths[pred_index]
                method_results = {
                    "path_all_levels_threshold": _path_threshold_match(gt_item["path"], pred_item["path"]),
                    "path_last_leaf_threshold": _path_last_leaf_threshold_match(gt_item["path"], pred_item["path"]),
                }
                if all(method_results.values()):
                    continue
                items.append(
                    {
                        "doc_id": document.doc_id,
                        "page_index": page.page_index,
                        "image_name": page.image_name,
                        "value": gt_item["value"],
                        "method_results": method_results,
                        "gt": gt_item,
                        "pred": pred_item,
                    }
                )

    payload = {
        "mistake_count": len(items),
        "matcher": "lcs",
        "matching_on": list(FACT_VALUE_EVALUATOR_SPEC.alignment_on) if FACT_VALUE_EVALUATOR_SPEC is not None else ["value"],
        "compare_methods": list(FACT_PATH_EVALUATOR_SPEC.compare_methods) if FACT_PATH_EVALUATOR_SPEC is not None and FACT_PATH_EVALUATOR_SPEC.compare_methods else ["path_all_levels_threshold"],
        "threshold": 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
                            "accuracy": f"{float(fact_result.details.get('value_accuracy', fact_result.exact_metrics.accuracy)):.6f}",
                            "helper_pair_count": fact_result.details.get("helper_pair_count", ""),
                            "string_similarity": "" if fact_result.string_similarity is None else f"{fact_result.string_similarity:.6f}",
                            "numeric_mae": "" if fact_result.numeric_mae is None else f"{fact_result.numeric_mae:.6f}",
                            "date_mae_days": "" if fact_result.date_mae_days is None else f"{fact_result.date_mae_days:.6f}",
                            "gt_value_preview": str(fact_result.details.get("gt_preview") or ""),
                            "pred_value_preview": str(fact_result.details.get("pred_preview") or ""),
                        }
                    )
