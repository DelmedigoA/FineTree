from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Literal


FactsMethod = Literal["count_only", "fact_quality_v1"]

FACT_FIELD_METHODS: dict[str, Literal["hard", "soft"]] = {
    "row_role": "hard",
    "comment_ref": "soft",
    "note_flag": "hard",
    "note_name": "soft",
    "note_num": "hard",
    "note_ref": "hard",
    "path_source": "hard",
    "currency": "hard",
    "scale": "hard",
    "value_type": "hard",
    "value_context": "hard",
}
FACT_SCORE_KEYS = ["facts_count_score", "date_score", *[f"{field_name}_score" for field_name in FACT_FIELD_METHODS]]
DATE_FIELDS = ["period_type", "period_start", "period_end", "duration_type"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def char_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def fuzzy_token_overlap_score(left: str, right: str) -> float:
    tokens_left = tokenize(left)
    tokens_right = tokenize(right)
    if not tokens_left or not tokens_right:
        return 0.0
    scores: list[float] = []
    used_indices: set[int] = set()
    for token_left in tokens_left:
        best_score = 0.0
        best_index: int | None = None
        for idx, token_right in enumerate(tokens_right):
            if idx in used_indices:
                continue
            score = char_similarity(token_left, token_right)
            if score > best_score:
                best_score = score
                best_index = idx
        if best_index is not None:
            used_indices.add(best_index)
            scores.append(best_score)
    denominator = max(len(tokens_left), len(tokens_right))
    return sum(scores) / float(denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def get_value_by_path(data: dict[str, Any], path: list[str | int]) -> Any:
    current: Any = data
    for part in path:
        if isinstance(part, int):
            if not isinstance(current, list) or part < 0 or part >= len(current):
                return None
            current = current[part]
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(str(part))
    return current


def path_to_str(path: list[str | int]) -> str:
    parts: list[str] = []
    for part in path:
        if isinstance(part, int):
            if not parts:
                parts.append(f"[{part}]")
            else:
                parts[-1] = f"{parts[-1]}[{part}]"
        else:
            parts.append(str(part))
    return ".".join(parts)


def hard_eq_score(left: dict[str, Any], right: dict[str, Any], path: list[str | int]) -> float:
    return float(get_value_by_path(left, path) == get_value_by_path(right, path))


def soft_eq_score(left: dict[str, Any], right: dict[str, Any], path: list[str | int]) -> float:
    value_left = get_value_by_path(left, path)
    value_right = get_value_by_path(right, path)
    if value_left == value_right:
        return 1.0
    if not isinstance(value_left, str) or not isinstance(value_right, str):
        return 0.0
    return fuzzy_token_overlap_score(value_left, value_right)


def hard_len_score(left: dict[str, Any], right: dict[str, Any], path: list[str | int]) -> float:
    value_left = get_value_by_path(left, path)
    value_right = get_value_by_path(right, path)
    if value_left is None or value_right is None:
        return 0.0
    try:
        return float(len(value_left) == len(value_right))
    except TypeError:
        return 0.0


def build_meta_field_report(
    pred_page: dict[str, Any],
    true_page: dict[str, Any],
    meta_rules: dict[str, dict[str, float]],
) -> dict[str, Any]:
    field_reports: dict[str, Any] = {}
    for path_name, weights in meta_rules.items():
        normalized_weights = {
            "hard": float(weights.get("hard", 0.0)),
            "soft": float(weights.get("soft", 0.0)),
        }
        path = path_name.split(".")
        field_name = path[-1]
        pred_value = get_value_by_path(pred_page, path)
        true_value = get_value_by_path(true_page, path)
        hard_score = hard_eq_score(pred_page, true_page, path)
        soft_score = soft_eq_score(pred_page, true_page, path)
        used_score = (normalized_weights["hard"] * hard_score) + (normalized_weights["soft"] * soft_score)
        field_reports[field_name] = {
            "path": path_to_str(path),
            "weights": normalized_weights,
            "pred_value": pred_value,
            "true_value": true_value,
            "hard_score": hard_score,
            "soft_score": soft_score,
            "used_score": used_score,
            "is_exact_match": pred_value == true_value,
            "error": None if pred_value == true_value else "value_mismatch",
        }
    return field_reports


def evaluate_meta(
    pred_page: dict[str, Any],
    true_page: dict[str, Any],
    meta_rules: dict[str, dict[str, float]],
) -> dict[str, Any]:
    field_reports = build_meta_field_report(pred_page, true_page, meta_rules)
    hard_scores = [info["hard_score"] for info in field_reports.values()]
    soft_scores = [info["used_score"] for info in field_reports.values()]
    return {
        "hard_score": _mean(hard_scores),
        "soft_score": _mean(soft_scores),
        "field_scores": {
            field_name: {
                "weights": info["weights"],
                "hard_score": info["hard_score"],
                "soft_score": info["soft_score"],
                "used_score": info["used_score"],
            }
            for field_name, info in field_reports.items()
        },
        "field_reports": field_reports,
    }


def _normalized_pred_facts(facts: Any) -> list[dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    normalized: list[dict[str, Any]] = []
    next_fact_num = 1
    for index, raw_fact in enumerate(facts, start=1):
        if not isinstance(raw_fact, dict):
            continue
        fact = dict(raw_fact)
        fact_num = fact.get("fact_num")
        if not isinstance(fact_num, int) or fact_num < 1:
            fact["fact_num"] = next_fact_num
        next_fact_num = max(next_fact_num, int(fact["fact_num"]) + 1)
        fact["_original_index"] = index - 1
        normalized.append(fact)
    return normalized


def _normalized_true_facts(facts: Any) -> list[dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw_fact in enumerate(facts, start=1):
        if not isinstance(raw_fact, dict):
            continue
        fact = dict(raw_fact)
        fact["_original_index"] = index - 1
        normalized.append(fact)
    return normalized


def _match_fact_pairs(
    pred_facts: list[dict[str, Any]],
    true_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    used_pred_indices: set[int] = set()
    pairs: list[dict[str, Any]] = []
    pred_index_by_fact_num: dict[int, int] = {}
    for pred_index, pred_fact in enumerate(pred_facts):
        fact_num = pred_fact.get("fact_num")
        if isinstance(fact_num, int) and fact_num >= 1 and fact_num not in pred_index_by_fact_num:
            pred_index_by_fact_num[fact_num] = pred_index

    for true_index, true_fact in enumerate(true_facts):
        matched_pred_index: int | None = None
        match_strategy = "unmatched_gt"
        true_fact_num = true_fact.get("fact_num")
        if isinstance(true_fact_num, int) and true_fact_num >= 1:
            candidate = pred_index_by_fact_num.get(true_fact_num)
            if candidate is not None and candidate not in used_pred_indices:
                matched_pred_index = candidate
                match_strategy = "fact_num"
        else:
            if true_index < len(pred_facts) and true_index not in used_pred_indices:
                matched_pred_index = true_index
                match_strategy = "page_order"
        if matched_pred_index is not None:
            used_pred_indices.add(matched_pred_index)
        pairs.append(
            {
                "pair_index": len(pairs),
                "match_strategy": match_strategy,
                "pred_index": matched_pred_index,
                "true_index": true_index,
                "pred_fact": pred_facts[matched_pred_index] if matched_pred_index is not None else None,
                "true_fact": true_fact,
            }
        )

    for pred_index, pred_fact in enumerate(pred_facts):
        if pred_index in used_pred_indices:
            continue
        pairs.append(
            {
                "pair_index": len(pairs),
                "match_strategy": "unmatched_pred",
                "pred_index": pred_index,
                "true_index": None,
                "pred_fact": pred_fact,
                "true_fact": None,
            }
        )
    return pairs


def _score_fact_field(
    pred_fact: dict[str, Any] | None,
    true_fact: dict[str, Any] | None,
    field_name: str,
    method: Literal["hard", "soft"],
) -> float:
    if pred_fact is None or true_fact is None:
        return 0.0
    if method == "hard":
        return hard_eq_score(pred_fact, true_fact, [field_name])
    return soft_eq_score(pred_fact, true_fact, [field_name])


def _score_fact_date(pred_fact: dict[str, Any] | None, true_fact: dict[str, Any] | None) -> float:
    if pred_fact is None or true_fact is None:
        return 0.0
    return float(all(hard_eq_score(pred_fact, true_fact, [field_name]) == 1.0 for field_name in DATE_FIELDS))


def evaluate_facts_count_only(pred_page: dict[str, Any], true_page: dict[str, Any]) -> dict[str, Any]:
    pred_facts = get_value_by_path(pred_page, ["facts"])
    true_facts = get_value_by_path(true_page, ["facts"])
    pred_len = len(pred_facts) if isinstance(pred_facts, list) else None
    true_len = len(true_facts) if isinstance(true_facts, list) else None
    facts_count_score = hard_len_score(pred_page, true_page, ["facts"])
    return {
        "facts_score": facts_count_score,
        "facts_count_score": facts_count_score,
        "pred_fact_count": pred_len,
        "true_fact_count": true_len,
        "count_difference": None if pred_len is None or true_len is None else pred_len - true_len,
        "is_exact_fact_count_match": pred_len == true_len,
        "error": None if pred_len == true_len else "fact_count_mismatch",
        "fact_scores": {},
        "fact_reports": [],
        "fact_pair_count": 0,
    }


def evaluate_facts_quality_v1(pred_page: dict[str, Any], true_page: dict[str, Any]) -> dict[str, Any]:
    pred_facts = _normalized_pred_facts(get_value_by_path(pred_page, ["facts"]))
    true_facts = _normalized_true_facts(get_value_by_path(true_page, ["facts"]))
    facts_count_score = float(len(pred_facts) == len(true_facts))
    fact_pairs = _match_fact_pairs(pred_facts, true_facts)

    fact_reports: list[dict[str, Any]] = []
    fact_metric_values: dict[str, list[float]] = {score_key: [] for score_key in FACT_SCORE_KEYS if score_key != "facts_count_score"}
    for pair in fact_pairs:
        pred_fact = pair["pred_fact"]
        true_fact = pair["true_fact"]
        field_scores: dict[str, float] = {}
        for field_name, method in FACT_FIELD_METHODS.items():
            score = _score_fact_field(pred_fact, true_fact, field_name, method)
            field_scores[f"{field_name}_score"] = score
            fact_metric_values[f"{field_name}_score"].append(score)
        date_score = _score_fact_date(pred_fact, true_fact)
        fact_metric_values["date_score"].append(date_score)
        fact_reports.append(
            {
                **pair,
                "date_score": date_score,
                "field_scores": field_scores,
            }
        )

    aggregate_scores = {
        "facts_count_score": facts_count_score,
        "date_score": _mean(fact_metric_values["date_score"]),
    }
    for field_name in FACT_FIELD_METHODS:
        aggregate_scores[f"{field_name}_score"] = _mean(fact_metric_values[f"{field_name}_score"])
    facts_score = _mean([aggregate_scores[key] for key in FACT_SCORE_KEYS])
    return {
        "facts_score": facts_score,
        **aggregate_scores,
        "pred_fact_count": len(pred_facts),
        "true_fact_count": len(true_facts),
        "count_difference": len(pred_facts) - len(true_facts),
        "is_exact_fact_count_match": len(pred_facts) == len(true_facts),
        "error": None if len(pred_facts) == len(true_facts) else "fact_count_mismatch",
        "fact_scores": {key: aggregate_scores[key] for key in aggregate_scores if key != "facts_count_score"},
        "fact_reports": fact_reports,
        "fact_pair_count": len(fact_pairs),
    }


def evaluate_facts(
    pred_page: dict[str, Any],
    true_page: dict[str, Any],
    *,
    facts_method: FactsMethod,
) -> dict[str, Any]:
    if facts_method == "count_only":
        return evaluate_facts_count_only(pred_page, true_page)
    if facts_method == "fact_quality_v1":
        return evaluate_facts_quality_v1(pred_page, true_page)
    raise ValueError(f"Unsupported facts method: {facts_method}")


def evaluate_page(
    pred_page: dict[str, Any],
    true_page: dict[str, Any],
    *,
    meta_rules: dict[str, dict[str, float]],
    aggregate_weights: dict[str, float],
    facts_method: FactsMethod,
) -> dict[str, Any]:
    meta_score = evaluate_meta(pred_page, true_page, meta_rules)
    facts_score = evaluate_facts(pred_page, true_page, facts_method=facts_method)
    overall_score = (
        aggregate_weights["meta_score"] * meta_score["soft_score"]
        + aggregate_weights["facts_score"] * facts_score["facts_score"]
    )
    return {
        "overall_score": overall_score,
        "meta_score": meta_score,
        "facts_score": facts_score,
    }


def build_page_debug_summary(page_report: dict[str, Any]) -> dict[str, Any]:
    meta = page_report["meta_score"]
    facts = page_report["facts_score"]
    meta_mismatches: list[dict[str, Any]] = []
    for field_name, info in meta["field_reports"].items():
        if not info["is_exact_match"]:
            meta_mismatches.append(
                {
                    "field": field_name,
                    "weights": info["weights"],
                    "hard_score": info["hard_score"],
                    "soft_score": info["soft_score"],
                    "used_score": info["used_score"],
                    "pred_value": info["pred_value"],
                    "true_value": info["true_value"],
                }
            )
    return {
        "overall_score": page_report["overall_score"],
        "meta_hard_score": meta["hard_score"],
        "meta_score": meta["soft_score"],
        "facts_score": facts["facts_score"],
        "meta_mismatches": meta_mismatches,
        "fact_count_debug": {
            "pred_fact_count": facts["pred_fact_count"],
            "true_fact_count": facts["true_fact_count"],
            "count_difference": facts["count_difference"],
            "is_exact_fact_count_match": facts["is_exact_fact_count_match"],
        },
        "fact_debug": {
            "facts_count_score": facts["facts_count_score"],
            "date_score": facts.get("date_score"),
            "fact_pair_count": facts.get("fact_pair_count", 0),
            "fact_reports": facts.get("fact_reports", []),
        },
    }


def _aggregate_meta_field_scores(page_reports: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = sorted(
        {
            field_name
            for page_report in page_reports
            for field_name in page_report["report"]["meta_score"]["field_scores"].keys()
        }
    )
    aggregated: dict[str, Any] = {}
    for field_name in field_names:
        field_entries = [
            page_report["report"]["meta_score"]["field_scores"][field_name]
            for page_report in page_reports
            if field_name in page_report["report"]["meta_score"]["field_scores"]
        ]
        aggregated[field_name] = {
            "hard_score": _mean([entry["hard_score"] for entry in field_entries]),
            "soft_score": _mean([entry["soft_score"] for entry in field_entries]),
            "used_score": _mean([entry["used_score"] for entry in field_entries]),
            "weights": field_entries[0]["weights"] if field_entries else {"hard": 0.0, "soft": 0.0},
        }
    return aggregated


def _aggregate_fact_scores(page_reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for page_report in page_reports
            for key in page_report["report"]["facts_score"].keys()
            if key.endswith("_score")
        }
    )
    return {
        key: _mean(
            [
                float(page_report["report"]["facts_score"][key])
                for page_report in page_reports
                if key in page_report["report"]["facts_score"]
            ]
        )
        for key in keys
    }


def evaluate_document_detailed(
    pred_doc: dict[str, Any],
    true_doc: dict[str, Any],
    *,
    meta_rules: dict[str, dict[str, float]],
    aggregate_weights: dict[str, float],
    facts_method: FactsMethod = "count_only",
) -> dict[str, Any]:
    pred_pages = pred_doc.get("pages")
    true_pages = true_doc.get("pages")
    if not isinstance(pred_pages, list) or not isinstance(true_pages, list):
        raise ValueError("Both prediction and ground-truth documents must contain a pages list.")
    doc_len_match = len(pred_pages) == len(true_pages)
    paired_count = min(len(pred_pages), len(true_pages))
    page_reports: list[dict[str, Any]] = []
    page_debug_summaries: list[dict[str, Any]] = []
    for page_index in range(paired_count):
        page_report = evaluate_page(
            pred_pages[page_index],
            true_pages[page_index],
            meta_rules=meta_rules,
            aggregate_weights=aggregate_weights,
            facts_method=facts_method,
        )
        page_reports.append(
            {
                "page_index": page_index,
                "overall_score": page_report["overall_score"],
                "report": page_report,
            }
        )
        page_debug_summaries.append({"page_index": page_index, **build_page_debug_summary(page_report)})
    field_scores = _aggregate_meta_field_scores(page_reports)
    fact_scores = _aggregate_fact_scores(page_reports)
    overall_score = _mean([page["overall_score"] for page in page_reports])
    meta_score = _mean([page["report"]["meta_score"]["soft_score"] for page in page_reports])
    meta_hard_score = _mean([page["report"]["meta_score"]["hard_score"] for page in page_reports])
    facts_score = _mean([page["report"]["facts_score"]["facts_score"] for page in page_reports])
    result = {
        "overall_score": overall_score,
        "document_score": overall_score,
        "meta_score": meta_score,
        "meta_hard_score": meta_hard_score,
        "facts_score": facts_score,
        "field_scores": field_scores,
        "doc_len_match": doc_len_match,
        "evaluated_page_count": paired_count,
        "page_debug_summaries": page_debug_summaries,
        "page_reports": page_reports,
    }
    result.update(fact_scores)
    return result


def aggregate_mapping_metrics(mapping_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not mapping_results:
        return {
            "overall_score": 0.0,
            "document_score": 0.0,
            "meta_score": 0.0,
            "meta_hard_score": 0.0,
            "facts_score": 0.0,
            "field_scores": {},
            "mapping_count": 0,
            "evaluated_page_count": 0,
            "doc_len_match_rate": 0.0,
            "entity_score": 0.0,
            "title_score": 0.0,
            "page_num_score": 0.0,
            "page_type_score": 0.0,
            "statement_type_score": 0.0,
            **{key: 0.0 for key in FACT_SCORE_KEYS if key != "facts_count_score"},
            "facts_count_score": 0.0,
        }
    metrics = [entry["metrics"] for entry in mapping_results]
    aggregated_field_scores: dict[str, Any] = {}
    field_names = sorted(
        {
            field_name
            for metric in metrics
            for field_name in metric.get("field_scores", {}).keys()
        }
    )
    for field_name in field_names:
        entries = [
            metric["field_scores"][field_name]
            for metric in metrics
            if field_name in metric.get("field_scores", {})
        ]
        aggregated_field_scores[field_name] = {
            "hard_score": _mean([entry["hard_score"] for entry in entries]),
            "soft_score": _mean([entry["soft_score"] for entry in entries]),
            "used_score": _mean([entry["used_score"] for entry in entries]),
            "weights": entries[0]["weights"] if entries else {"hard": 0.0, "soft": 0.0},
        }
    field_metric_names = {
        "entity_name": "entity_score",
        "title": "title_score",
        "page_num": "page_num_score",
        "page_type": "page_type_score",
        "statement_type": "statement_type_score",
    }
    aggregate = {
        "overall_score": _mean([metric["overall_score"] for metric in metrics]),
        "document_score": _mean([metric["document_score"] for metric in metrics]),
        "meta_score": _mean([metric["meta_score"] for metric in metrics]),
        "meta_hard_score": _mean([metric["meta_hard_score"] for metric in metrics]),
        "facts_score": _mean([metric["facts_score"] for metric in metrics]),
        "field_scores": aggregated_field_scores,
        "mapping_count": len(mapping_results),
        "evaluated_page_count": int(sum(int(metric["evaluated_page_count"]) for metric in metrics)),
        "doc_len_match_rate": _mean([1.0 if metric["doc_len_match"] else 0.0 for metric in metrics]),
    }
    fact_metric_names = sorted(
        {
            key
            for metric in metrics
            for key in metric.keys()
            if key.endswith("_score")
            and key not in {"overall_score", "document_score", "meta_score", "meta_hard_score", "facts_score"}
        }
    )
    for metric_name in fact_metric_names:
        aggregate[metric_name] = _mean([float(metric.get(metric_name, 0.0)) for metric in metrics])
    for field_name, metric_name in field_metric_names.items():
        aggregate[metric_name] = float(aggregated_field_scores.get(field_name, {}).get("used_score", 0.0))
    return aggregate
