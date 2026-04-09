from __future__ import annotations

from statistics import mean
from typing import Any

from ..algorithms import (
    align_sequences_by_index,
    build_row_diff_diagnostics,
    date_diff_details,
    lcs_metrics,
    numeric_closeness_details,
    sequence_ratio,
)
from ..io.ground_truth import load_ground_truth_page
from ..models import DocumentResult, FactFieldResult, MetaFieldResult, PageResult, RunResult
from ..spec import DATE_FACT_FIELDS, FACT_FIELDS, META_FIELDS, SOFT_STRING_FACT_FIELDS, SPARSE_OPTIONAL_FACT_FIELDS, joined_path, normalize_fact_value, normalize_meta_value, parse_date_value, parse_numeric_value, primary_fact_fields


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _mean_available(values: list[float | None]) -> float:
    available = [float(value) for value in values if value is not None]
    return _mean(available)


def _sequence_preview(values: list[Any], *, max_items: int = 3, max_length: int = 120) -> str:
    if not values:
        return ""
    rendered: list[str] = []
    for value in values[:max_items]:
        if isinstance(value, list):
            rendered.append(joined_path(value))
        elif value is None:
            rendered.append("null")
        else:
            rendered.append(str(value))
    text = " | ".join(rendered)
    if len(values) > max_items:
        text += f" | ... (+{len(values) - max_items})"
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _populated_sequence_preview(values: list[Any], *, max_items: int = 3, max_length: int = 120) -> str:
    populated = [value for value in values if value not in (None, [], "")]
    return _sequence_preview(populated, max_items=max_items, max_length=max_length)


def _meta_result(gt_meta: dict[str, Any], pred_meta: dict[str, Any]) -> tuple[dict[str, MetaFieldResult], float]:
    results: dict[str, MetaFieldResult] = {}
    scores: list[float] = []
    for spec in META_FIELDS:
        gt_value = normalize_meta_value(spec.field, gt_meta.get(spec.field))
        pred_value = normalize_meta_value(spec.field, pred_meta.get(spec.field))
        if spec.comparison_type == "hard":
            score = 1.0 if gt_value == pred_value else 0.0
        else:
            score = sequence_ratio("" if gt_value is None else str(gt_value), "" if pred_value is None else str(pred_value))
        scores.append(score)
        results[spec.field] = MetaFieldResult(
            field=spec.field,
            comparison_type=spec.comparison_type,
            gt_value=gt_value,
            pred_value=pred_value,
            score=float(score),
            is_exact_match=gt_value == pred_value,
        )
    return results, _mean(scores)


def _field_sequence(facts: list[dict[str, Any]], field_name: str) -> list[Any]:
    return [normalize_fact_value(field_name, fact.get(field_name)) for fact in facts if isinstance(fact, dict)]


def _row_keys(facts: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        keys.append(
            " | ".join(
                [
                    str(fact.get("fact_num")),
                    joined_path(normalize_fact_value("path", fact.get("path"))),
                    str(normalize_fact_value("value", fact.get("value"))),
                ]
            )
        )
    return keys


def _fact_field_result(field_name: str, gt_facts: list[dict[str, Any]], pred_facts: list[dict[str, Any]]) -> FactFieldResult:
    gt_sequence = _field_sequence(gt_facts, field_name)
    pred_sequence = _field_sequence(pred_facts, field_name)
    exact = lcs_metrics(gt_sequence, pred_sequence)
    result = FactFieldResult(field=field_name, score=float(exact.f1), exact_metrics=exact)
    if field_name in SPARSE_OPTIONAL_FACT_FIELDS:
        result.details["gt_preview"] = _populated_sequence_preview(gt_sequence)
        result.details["pred_preview"] = _populated_sequence_preview(pred_sequence)
        result.details["gt_non_null_count"] = sum(1 for value in gt_sequence if value not in (None, [], ""))
        result.details["pred_non_null_count"] = sum(1 for value in pred_sequence if value not in (None, [], ""))
        result.details["gt_null_count"] = len(gt_sequence) - int(result.details["gt_non_null_count"])
        result.details["pred_null_count"] = len(pred_sequence) - int(result.details["pred_non_null_count"])
        result.details["gt_preview_populated"] = result.details["gt_preview"]
        result.details["pred_preview_populated"] = result.details["pred_preview"]
    else:
        result.details["gt_preview"] = _sequence_preview(gt_sequence)
        result.details["pred_preview"] = _sequence_preview(pred_sequence)
    result.details["score_components"] = {"exact_f1": float(exact.f1), "formula": "exact_f1"}

    if field_name == "value":
        numeric_pairs = align_sequences_by_index(
            [parse_numeric_value(value) for value in gt_sequence],
            [parse_numeric_value(value) for value in pred_sequence],
        )
        numeric = numeric_closeness_details(numeric_pairs)
        numeric_score = numeric["score"]
        result.numeric_mae = numeric["mae"]
        result.details["numeric_closeness_score"] = numeric_score
        result.details["helper_pair_count"] = int(numeric["pair_count"])
        if numeric_score is None:
            result.score = float(exact.f1)
            result.details["score_components"] = {"exact_f1": float(exact.f1), "formula": "exact_f1"}
        else:
            result.score = _mean_available([exact.f1, float(numeric_score)])
            result.details["score_components"] = {
                "exact_f1": float(exact.f1),
                "numeric_closeness_score": float(numeric_score),
                "formula": "mean(exact_f1, numeric_closeness_score)",
            }
        return result

    if field_name == "path":
        if not gt_sequence and not pred_sequence:
            result.string_similarity = 1.0
            result.details["joined_path_similarity"] = 1.0
            result.details["elementwise_path_similarity"] = 1.0
            result.details["helper_pair_count"] = 0
            result.score = 1.0
            result.details["score_components"] = {
                "exact_f1": float(exact.f1),
                "joined_path_similarity": 1.0,
                "elementwise_path_similarity": 1.0,
                "formula": "mean(exact_f1, joined_path_similarity, elementwise_path_similarity)",
            }
            return result
        joined_pairs = align_sequences_by_index(
            [joined_path(value) for value in gt_sequence],
            [joined_path(value) for value in pred_sequence],
            missing_sentinel="",
        )
        joined_similarity = _mean([sequence_ratio(str(gt_value), str(pred_value)) for gt_value, pred_value in joined_pairs])
        element_scores: list[float] = []
        for gt_value, pred_value in align_sequences_by_index(gt_sequence, pred_sequence, missing_sentinel=[]):
            left = list(gt_value or [])
            right = list(pred_value or [])
            max_len = max(len(left), len(right))
            if max_len == 0:
                element_scores.append(1.0)
                continue
            row_scores = []
            for index in range(max_len):
                left_token = left[index] if index < len(left) else "<missing>"
                right_token = right[index] if index < len(right) else "<missing>"
                row_scores.append(sequence_ratio(str(left_token), str(right_token)))
            element_scores.append(_mean(row_scores))
        result.string_similarity = joined_similarity
        result.details["joined_path_similarity"] = joined_similarity
        result.details["elementwise_path_similarity"] = _mean(element_scores)
        result.details["helper_pair_count"] = len(joined_pairs)
        result.score = _mean([exact.f1, joined_similarity, _mean(element_scores)])
        result.details["score_components"] = {
            "exact_f1": float(exact.f1),
            "joined_path_similarity": float(joined_similarity),
            "elementwise_path_similarity": float(_mean(element_scores)),
            "formula": "mean(exact_f1, joined_path_similarity, elementwise_path_similarity)",
        }
        return result

    if field_name in DATE_FACT_FIELDS:
        date_pairs = align_sequences_by_index(
            [parse_date_value(value) for value in gt_sequence],
            [parse_date_value(value) for value in pred_sequence],
        )
        date_details = date_diff_details(date_pairs)
        date_score = date_details["score"]
        result.date_mae_days = date_details["mae_days"]
        result.details["date_diff_score"] = date_score
        result.details["helper_pair_count"] = int(date_details["pair_count"])
        if date_score is None:
            result.score = float(exact.f1)
            result.details["score_components"] = {"exact_f1": float(exact.f1), "formula": "exact_f1"}
        else:
            result.score = _mean_available([exact.f1, float(date_score)])
            result.details["score_components"] = {
                "exact_f1": float(exact.f1),
                "date_diff_score": float(date_score),
                "formula": "mean(exact_f1, date_diff_score)",
            }
        return result

    if field_name in SOFT_STRING_FACT_FIELDS:
        string_pairs = align_sequences_by_index(gt_sequence, pred_sequence, missing_sentinel="")
        result.string_similarity = _mean(
            [sequence_ratio("" if gt_value is None else str(gt_value), "" if pred_value is None else str(pred_value)) for gt_value, pred_value in string_pairs]
        )
        result.details["helper_pair_count"] = len(string_pairs)
        result.details["score_components"] = {"exact_f1": float(exact.f1), "string_similarity": float(result.string_similarity), "formula": "exact_f1"}
        return result

    return result


def evaluate_single_page(doc_id: str, page_index: int, gt_page: dict[str, Any], pred_page: dict[str, Any]) -> PageResult:
    gt_meta = gt_page.get("meta") if isinstance(gt_page.get("meta"), dict) else {}
    pred_meta = pred_page.get("meta") if isinstance(pred_page.get("meta"), dict) else {}
    meta_result, meta_score = _meta_result(gt_meta, pred_meta)

    gt_facts = gt_page.get("facts") if isinstance(gt_page.get("facts"), list) else []
    pred_facts = pred_page.get("facts") if isinstance(pred_page.get("facts"), list) else []
    gt_fact_count = len(gt_facts)
    pred_fact_count = len(pred_facts)
    facts_result = {
        spec.field: _fact_field_result(spec.field, gt_facts, pred_facts)
        for spec in FACT_FIELDS
    }
    primary_scores = [facts_result[spec.field].score for spec in primary_fact_fields()]
    facts_applicable = True
    if gt_fact_count == 0 and pred_fact_count == 0:
        facts_status = "empty_match"
        facts_score = 1.0
    elif gt_fact_count == 0 and pred_fact_count > 0:
        facts_status = "false_positive_on_empty_gt"
        facts_score = float(1.0 / (1.0 + pred_fact_count))
    else:
        facts_status = "standard"
        facts_score = _mean(primary_scores)
    page_score = _mean([meta_score, facts_score])
    image_name = str(pred_page.get("image") or gt_page.get("image") or f"page_{page_index:04d}.png")

    return PageResult(
        doc_id=doc_id,
        page_index=page_index,
        image_name=image_name,
        meta_result=meta_result,
        facts_result=facts_result,
        gt_fact_count=gt_fact_count,
        pred_fact_count=pred_fact_count,
        facts_applicable=facts_applicable,
        facts_status=facts_status,
        meta_score=meta_score,
        facts_score=facts_score,
        page_score=page_score,
        row_diff_diagnostics=build_row_diff_diagnostics(_row_keys(gt_facts), _row_keys(pred_facts)),
    )


def evaluate_predictions_bundle(bundle: dict[str, Any], *, data_root: Any) -> RunResult:
    document_results: list[DocumentResult] = []
    for doc_id in sorted(bundle.get("documents", {}).keys()):
        prediction_document = bundle["documents"][doc_id]
        page_results: list[PageResult] = []
        for page_entry in prediction_document.get("pages", []):
            page_index = int(page_entry.get("page_index") or 0)
            if page_index < 1:
                continue
            gt_page = load_ground_truth_page(doc_id, page_index, data_root=data_root)
            pred_page = page_entry.get("parsed_page") if isinstance(page_entry.get("parsed_page"), dict) else {}
            page_results.append(evaluate_single_page(doc_id, page_index, gt_page, pred_page))
        if not page_results:
            continue
        document_results.append(
            DocumentResult(
                doc_id=doc_id,
                meta_score=_mean([page.meta_score for page in page_results]),
                facts_score=_mean([page.facts_score for page in page_results]),
                document_score=_mean([page.page_score for page in page_results]),
                pages=page_results,
            )
        )
    return RunResult(
        run_score=_mean([document.document_score for document in document_results]),
        meta_score=_mean([document.meta_score for document in document_results]),
        facts_score=_mean([document.facts_score for document in document_results]),
        documents=document_results,
    )
