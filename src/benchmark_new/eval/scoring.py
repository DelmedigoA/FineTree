from __future__ import annotations

from statistics import mean
from typing import Any

from ..algorithms import (
    align_sequences_by_index,
    build_row_diff_diagnostics,
    date_diff_details,
    lcs_index_pairs,
    lcs_metrics,
    numeric_closeness_details,
    sequence_ratio,
)
from ..evaluation_specs import get_facts_field_spec
from ..io.ground_truth import load_ground_truth_page
from ..models import DocumentResult, FactFieldResult, MetaFieldResult, PageResult, RunResult
from ..spec import DATE_FACT_FIELDS, FACT_FIELDS, META_FIELDS, SOFT_STRING_FACT_FIELDS, SPARSE_OPTIONAL_FACT_FIELDS, joined_path, normalize_fact_value, normalize_meta_value, parse_date_value, parse_numeric_value, primary_fact_fields


FACT_VALUE_EVALUATOR_SPEC = get_facts_field_spec("value")
FACT_PATH_EVALUATOR_SPEC = get_facts_field_spec("path")


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


def evaluate_page_meta(gt_meta: dict[str, Any], pred_meta: dict[str, Any]) -> tuple[dict[str, MetaFieldResult], float]:
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


def _matched_pair_accuracy(
    gt_values: list[Any],
    pred_values: list[Any],
    *,
    matched_pairs: list[tuple[int, int]],
) -> float:
    if not matched_pairs:
        return 1.0 if not gt_values and not pred_values else 0.0
    matches = 0
    for gt_index, pred_index in matched_pairs:
        if gt_values[gt_index] == pred_values[pred_index]:
            matches += 1
    return float(matches / len(matched_pairs))


def _scores_from_matched_pairs(
    matched_pairs: list[tuple[int, int]],
    *,
    gt_values: list[Any],
    pred_values: list[Any],
    scorer: Any,
) -> list[float]:
    matched_scores = [float(scorer(gt_values[gt_index], pred_values[pred_index])) for gt_index, pred_index in matched_pairs]
    unmatched_count = max(len(gt_values), len(pred_values)) - len(matched_pairs)
    return matched_scores + ([0.0] * max(unmatched_count, 0))


def _fact_value_matched_pairs(gt_facts: list[dict[str, Any]], pred_facts: list[dict[str, Any]]) -> list[tuple[int, int]]:
    gt_values = _field_sequence(gt_facts, "value")
    pred_values = _field_sequence(pred_facts, "value")
    return lcs_index_pairs(gt_values, pred_values)


def _values_from_pairs(
    matched_pairs: list[tuple[int, int]],
    *,
    gt_values: list[Any],
    pred_values: list[Any],
) -> list[tuple[Any, Any]]:
    return [(gt_values[gt_index], pred_values[pred_index]) for gt_index, pred_index in matched_pairs]


def _value_accuracy_scope() -> str:
    if FACT_VALUE_EVALUATOR_SPEC is None:
        return "matched_pairs"
    group = FACT_VALUE_EVALUATOR_SPEC.metric_groups.get("value")
    if group is None or group.on in (None, ""):
        return "matched_pairs"
    return str(group.on)


def _path_all_levels_threshold_match(gt_path: Any, pred_path: Any) -> float:
    left = list(gt_path or [])
    right = list(pred_path or [])
    threshold = 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
    require_same_length = True if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.require_same_length is None else bool(FACT_PATH_EVALUATOR_SPEC.require_same_length)
    if require_same_length and len(left) != len(right):
        return 0.0
    if len(left) != len(right):
        return 0.0
    for left_item, right_item in zip(left, right):
        if sequence_ratio(str(left_item), str(right_item)) < threshold:
            return 0.0
    return 1.0


def _path_last_leaf_threshold_match(gt_path: Any, pred_path: Any) -> float:
    left = list(gt_path or [])
    right = list(pred_path or [])
    threshold = 0.85 if FACT_PATH_EVALUATOR_SPEC is None or FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
    if not left or not right:
        return 0.0
    return 1.0 if sequence_ratio(str(left[-1]), str(right[-1])) >= threshold else 0.0


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


def _binary_metrics_from_aligned_scores(
    scores: list[float],
    *,
    gt_count: int,
    pred_count: int,
) -> tuple["FieldExactMetrics", float]:
    matches = int(sum(1 for score in scores if score >= 1.0))
    if gt_count == 0 and pred_count == 0:
        precision = 1.0
        recall = 1.0
    elif pred_count == 0:
        precision = 1.0
        recall = 0.0
    elif gt_count == 0:
        precision = 0.0
        recall = 1.0
    else:
        precision = float(matches / pred_count)
        recall = float(matches / gt_count)
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    denominator = max(gt_count, pred_count)
    accuracy = float(sum(scores) / denominator) if denominator else 1.0
    from ..models import FieldExactMetrics

    return (
        FieldExactMetrics(
            matches=matches,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
        ),
        accuracy,
    )


def _fact_field_result(field_name: str, gt_facts: list[dict[str, Any]], pred_facts: list[dict[str, Any]]) -> FactFieldResult:
    gt_sequence = _field_sequence(gt_facts, field_name)
    pred_sequence = _field_sequence(pred_facts, field_name)
    if field_name == "value":
        if FACT_VALUE_EVALUATOR_SPEC is not None and FACT_VALUE_EVALUATOR_SPEC.matcher not in {None, "lcs"}:
            raise ValueError(f"Unsupported facts.value matcher: {FACT_VALUE_EVALUATOR_SPEC.matcher}")
        if FACT_VALUE_EVALUATOR_SPEC is not None and FACT_VALUE_EVALUATOR_SPEC.alignment_on not in {(), ("value",)}:
            raise ValueError(f"Unsupported facts.value alignment_on: {FACT_VALUE_EVALUATOR_SPEC.alignment_on}")
        exact = lcs_metrics(gt_sequence, pred_sequence)
        matched_pairs = lcs_index_pairs(gt_sequence, pred_sequence)
    else:
        matched_pairs = _fact_value_matched_pairs(gt_facts, pred_facts)
        exact_scores = _scores_from_matched_pairs(
            matched_pairs,
            gt_values=gt_sequence,
            pred_values=pred_sequence,
            scorer=lambda gt_value, pred_value: 1.0 if gt_value == pred_value else 0.0,
        )
        exact, _ = _binary_metrics_from_aligned_scores(exact_scores, gt_count=len(gt_sequence), pred_count=len(pred_sequence))
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
        value_accuracy_scope = _value_accuracy_scope()
        if value_accuracy_scope != "matched_pairs":
            raise ValueError(f"Unsupported facts.value accuracy scope: {value_accuracy_scope}")
        value_accuracy = _matched_pair_accuracy(gt_sequence, pred_sequence, matched_pairs=matched_pairs)
        numeric_pairs = align_sequences_by_index(
            [parse_numeric_value(value) for value in gt_sequence],
            [parse_numeric_value(value) for value in pred_sequence],
        )
        numeric = numeric_closeness_details(numeric_pairs)
        numeric_score = numeric["score"]
        result.numeric_mae = numeric["mae"]
        result.details["value_accuracy"] = value_accuracy
        result.details["alignment_precision"] = float(exact.precision)
        result.details["alignment_recall"] = float(exact.recall)
        result.details["alignment_f1"] = float(exact.f1)
        result.details["matched_pair_count"] = len(matched_pairs)
        result.details["alignment_matcher"] = "lcs"
        result.details["value_accuracy_scope"] = value_accuracy_scope
        result.details["numeric_closeness_score"] = numeric_score
        result.details["helper_pair_count"] = int(numeric["pair_count"])
        result.score = _mean([float(exact.f1), float(value_accuracy)])
        result.details["score_components"] = {
            "alignment_f1": float(exact.f1),
            "value_accuracy": float(value_accuracy),
            "formula": "mean(alignment_f1, value_accuracy)",
        }
        return result

    if field_name == "path":
        path_methods = ("path_all_levels_threshold",) if FACT_PATH_EVALUATOR_SPEC is None or not FACT_PATH_EVALUATOR_SPEC.compare_methods else FACT_PATH_EVALUATOR_SPEC.compare_methods
        if FACT_PATH_EVALUATOR_SPEC is not None and FACT_PATH_EVALUATOR_SPEC.compare_method in {"path_all_levels_threshold", "path_last_leaf_threshold", "path_soft_overlap"}:
            value_gt_sequence = _field_sequence(gt_facts, "value")
            value_pred_sequence = _field_sequence(pred_facts, "value")
            matched_pairs = lcs_index_pairs(value_gt_sequence, value_pred_sequence)
            method_scores: dict[str, float] = {}
            method_matches: dict[str, int] = {}
            method_summaries: dict[str, dict[str, float | int]] = {}
            exact_metrics_for_primary = result.exact_metrics
            primary_method = path_methods[0]
            for method_name in path_methods:
                if method_name == "path_all_levels_threshold":
                    scorer = _path_all_levels_threshold_match
                    pair_scores = _scores_from_matched_pairs(
                        matched_pairs,
                        gt_values=gt_sequence,
                        pred_values=pred_sequence,
                        scorer=scorer,
                    )
                    metrics, accuracy = _binary_metrics_from_aligned_scores(pair_scores, gt_count=len(gt_sequence), pred_count=len(pred_sequence))
                    method_scores[method_name] = accuracy
                    method_matches[method_name] = int(metrics.matches)
                    method_summaries[method_name] = {
                        "accuracy": float(accuracy),
                        "matches": int(metrics.matches),
                        "precision": float(metrics.precision),
                        "recall": float(metrics.recall),
                        "f1": float(metrics.f1),
                    }
                elif method_name == "path_last_leaf_threshold":
                    scorer = _path_last_leaf_threshold_match
                    pair_scores = _scores_from_matched_pairs(
                        matched_pairs,
                        gt_values=gt_sequence,
                        pred_values=pred_sequence,
                        scorer=scorer,
                    )
                    metrics, accuracy = _binary_metrics_from_aligned_scores(pair_scores, gt_count=len(gt_sequence), pred_count=len(pred_sequence))
                    method_scores[method_name] = accuracy
                    method_matches[method_name] = int(metrics.matches)
                    method_summaries[method_name] = {
                        "accuracy": float(accuracy),
                        "matches": int(metrics.matches),
                        "precision": float(metrics.precision),
                        "recall": float(metrics.recall),
                        "f1": float(metrics.f1),
                    }
                elif method_name == "path_soft_overlap":
                    pair_metrics = [_path_soft_overlap_metrics(gt_sequence[gt_index], pred_sequence[pred_index]) for gt_index, pred_index in matched_pairs]
                    shared_nodes = sum(int(item["shared_nodes"]) for item in pair_metrics)
                    gt_nodes = sum(len(list(gt_sequence[gt_index] or [])) for gt_index, _ in matched_pairs)
                    pred_nodes = sum(len(list(pred_sequence[pred_index] or [])) for _, pred_index in matched_pairs)
                    precision = float(shared_nodes / pred_nodes) if pred_nodes else 1.0
                    recall = float(shared_nodes / gt_nodes) if gt_nodes else 1.0
                    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
                    method_scores[method_name] = f1
                    method_matches[method_name] = int(shared_nodes)
                    method_summaries[method_name] = {
                        "shared_nodes": int(shared_nodes),
                        "gt_nodes": int(gt_nodes),
                        "pred_nodes": int(pred_nodes),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                    if method_name == primary_method:
                        exact_metrics_for_primary = exact_metrics_for_primary
                    continue
                else:
                    raise ValueError(f"Unsupported facts.path compare method: {method_name}")
                if method_name == primary_method:
                    exact_metrics_for_primary = metrics
            result.exact_metrics = exact_metrics_for_primary
            result.details["helper_pair_count"] = len(matched_pairs)
            result.details["matching_on"] = "value"
            result.details["threshold"] = 0.85 if FACT_PATH_EVALUATOR_SPEC.threshold is None else float(FACT_PATH_EVALUATOR_SPEC.threshold)
            result.details["require_same_length"] = True if FACT_PATH_EVALUATOR_SPEC.require_same_length is None else bool(FACT_PATH_EVALUATOR_SPEC.require_same_length)
            result.details["path_method_scores"] = method_scores
            result.details["path_method_summaries"] = method_summaries
            result.score = _mean(list(method_scores.values()))
            result.details["score_components"] = {
                **{method_name: float(score) for method_name, score in method_scores.items()},
                "formula": "mean(path method accuracies)",
            }
            return result
        return result

    if field_name in DATE_FACT_FIELDS:
        date_pairs = _values_from_pairs(
            matched_pairs,
            gt_values=[parse_date_value(value) for value in gt_sequence],
            pred_values=[parse_date_value(value) for value in pred_sequence],
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
        string_pairs = _values_from_pairs(matched_pairs, gt_values=gt_sequence, pred_values=pred_sequence)
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
    meta_result, meta_score = evaluate_page_meta(gt_meta, pred_meta)

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
