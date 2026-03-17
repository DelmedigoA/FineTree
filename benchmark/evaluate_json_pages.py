#!/usr/bin/env python3
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import yaml


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_page_obj(obj):
    """
    Accept either:
    1. a single page object with keys: meta, facts
    2. a document object with a single page in obj["pages"]
    """
    if isinstance(obj, dict) and "pages" in obj:
        pages = obj.get("pages") or []
        if len(pages) == 1:
            return pages[0]
    return obj


def ensure_fact_nums(page):
    page = deepcopy(page)
    facts = page.get("facts", []) or []
    for i, fact in enumerate(facts, start=1):
        fact.setdefault("fact_num", i)
    return page


def normalize_string(value):
    if value is None:
        return None
    return str(value).strip()


def normalize_casefold(value):
    if value is None:
        return None
    return str(value).strip().casefold()


def maybe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def compare_values(pred_value, gt_value, method="equality", params=None):
    params = params or {}

    if method == "equality":
        return 1.0 if pred_value == gt_value else 0.0

    if method == "string_normalized":
        return 1.0 if normalize_string(pred_value) == normalize_string(gt_value) else 0.0

    if method == "string_casefold":
        return 1.0 if normalize_casefold(pred_value) == normalize_casefold(gt_value) else 0.0

    if method == "contains":
        p = normalize_casefold(pred_value)
        g = normalize_casefold(gt_value)
        if p is None or g is None:
            return 1.0 if p == g else 0.0
        return 1.0 if (p in g or g in p) else 0.0

    if method == "numeric":
        p = maybe_float(pred_value)
        g = maybe_float(gt_value)
        if p is None or g is None:
            return 0.0
        tol = float(params.get("tolerance", 0.0))
        return 1.0 if abs(p - g) <= tol else 0.0

    if method == "list_exact":
        return 1.0 if pred_value == gt_value else 0.0

    if method == "path_exact":
        p = [normalize_string(x) for x in (pred_value or [])]
        g = [normalize_string(x) for x in (gt_value or [])]
        return 1.0 if p == g else 0.0

    if method == "bbox_tolerance":
        p = pred_value or []
        g = gt_value or []
        if not isinstance(p, list) or not isinstance(g, list) or len(p) != 4 or len(g) != 4:
            return 0.0
        tol = float(params.get("tolerance", 0.0))
        try:
            ok = all(abs(float(a) - float(b)) <= tol for a, b in zip(p, g))
        except Exception:
            return 0.0
        return 1.0 if ok else 0.0

    if method == "equations_exact":
        p = pred_value or []
        g = gt_value or []
        return 1.0 if p == g else 0.0

    raise ValueError(f"Unknown comparison method: {method}")


def build_fact_map(facts, key_name="fact_num"):
    out = {}
    for i, fact in enumerate(facts, start=1):
        key = fact.get(key_name, i)
        out[key] = fact
    return out


def resolve_jobs(config):
    dataset = config["dataset"]

    if dataset.get("page_mappings"):
        jobs = []
        for item in dataset["page_mappings"]:
            jobs.append(
                {
                    "idx": int(item["idx"]),
                    "pred_path": item["pred_path"],
                }
            )
        return jobs

    start_idx = int(dataset["start_idx"])
    end_idx = int(dataset["end_idx"])
    pred_pattern = dataset["pred_pattern"]

    jobs = []
    for idx in range(start_idx, end_idx + 1):
        jobs.append(
            {
                "idx": idx,
                "pred_path": pred_pattern.format(idx=idx),
            }
        )
    return jobs


def score_field(pred_value, gt_value, spec):
    method = spec.get("method", "equality")
    weight = float(spec.get("weight", 1.0))
    params = spec.get("params", {})
    score = compare_values(pred_value, gt_value, method=method, params=params)
    return score, weight


def evaluate_page(pred_page, gt_page, config):
    pred_page = ensure_fact_nums(normalize_page_obj(pred_page))
    gt_page = ensure_fact_nums(normalize_page_obj(gt_page))

    eval_cfg = config["evaluation"]
    facts_key = eval_cfg.get("facts_key", "fact_num")
    skip_0 = bool(eval_cfg.get("skip_0", True))
    skip_mismatch = bool(eval_cfg.get("skip_mismatch", True))

    pred_facts = pred_page.get("facts", []) or []
    gt_facts = gt_page.get("facts", []) or []

    if len(gt_facts) == 0 and skip_0:
        return {
            "status": "skipped_zero_gt_facts",
            "page_score": None,
            "details": {},
        }

    if len(pred_facts) == 0 and skip_0:
        return {
            "status": "skipped_zero_pred_facts",
            "page_score": None,
            "details": {},
        }

    if len(pred_facts) != len(gt_facts) and skip_mismatch:
        return {
            "status": "skipped_mismatch",
            "page_score": None,
            "details": {
                "pred_fact_count": len(pred_facts),
                "gt_fact_count": len(gt_facts),
            },
        }

    details = {"meta": {}, "facts": {}}
    weighted_sum = 0.0
    total_weight = 0.0

    # Meta
    pred_meta = pred_page.get("meta", {}) or {}
    gt_meta = gt_page.get("meta", {}) or {}

    for field_name, spec in config.get("meta_fields", {}).items():
        pred_value = pred_meta.get(field_name)
        gt_value = gt_meta.get(field_name)
        score, weight = score_field(pred_value, gt_value, spec)

        details["meta"][field_name] = {
            "pred_value": pred_value,
            "gt_value": gt_value,
            "score": score,
            "weight": weight,
            "method": spec.get("method", "equality"),
        }

        weighted_sum += score * weight
        total_weight += weight

    # Facts
    pred_map = build_fact_map(pred_facts, key_name=facts_key)
    gt_map = build_fact_map(gt_facts, key_name=facts_key)

    all_fact_keys = sorted(set(pred_map.keys()) | set(gt_map.keys()), key=lambda x: (isinstance(x, str), x))

    for fact_key in all_fact_keys:
        pred_fact = pred_map.get(fact_key)
        gt_fact = gt_map.get(fact_key)

        details["facts"][str(fact_key)] = {}

        for field_name, spec in config.get("fact_fields", {}).items():
            pred_value = None if pred_fact is None else pred_fact.get(field_name)
            gt_value = None if gt_fact is None else gt_fact.get(field_name)
            score, weight = score_field(pred_value, gt_value, spec)

            details["facts"][str(fact_key)][field_name] = {
                "pred_value": pred_value,
                "gt_value": gt_value,
                "score": score,
                "weight": weight,
                "method": spec.get("method", "equality"),
            }

            weighted_sum += score * weight
            total_weight += weight

    page_score = None if total_weight == 0 else weighted_sum / total_weight

    return {
        "status": "evaluated",
        "page_score": page_score,
        "details": details,
        "pred_fact_count": len(pred_facts),
        "gt_fact_count": len(gt_facts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted page JSON files against GT page JSON.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    dataset = config["dataset"]
    gt_doc = load_json(dataset["gt_path"])
    gt_pages = gt_doc.get("pages", [])

    jobs = resolve_jobs(config)
    report = {
        "config_path": str(Path(args.config).resolve()),
        "gt_path": dataset["gt_path"],
        "pages": [],
        "summary": {},
    }

    page_scores = []

    for job in jobs:
        idx = job["idx"]
        pred_path = job["pred_path"]

        if idx < 1 or idx > len(gt_pages):
            report["pages"].append(
                {
                    "idx": idx,
                    "pred_path": pred_path,
                    "status": "invalid_gt_page_index",
                }
            )
            print(f"[{idx}] invalid GT page index")
            continue

        try:
            pred_obj = load_json(pred_path)
        except Exception as e:
            report["pages"].append(
                {
                    "idx": idx,
                    "pred_path": pred_path,
                    "status": "pred_load_error",
                    "error": str(e),
                }
            )
            print(f"[{idx}] prediction load error: {e}")
            continue

        gt_page = gt_pages[idx - 1]
        page_result = evaluate_page(pred_obj, gt_page, config)
        page_result["idx"] = idx
        page_result["pred_path"] = pred_path
        report["pages"].append(page_result)

        status = page_result["status"]
        score = page_result["page_score"]

        if score is not None:
            page_scores.append(score)
            print(f"[{idx}] {status} | score={score:.4f}")
        else:
            print(f"[{idx}] {status}")

    report["summary"] = {
        "evaluated_pages": sum(1 for p in report["pages"] if p.get("status") == "evaluated"),
        "skipped_pages": sum(1 for p in report["pages"] if str(p.get("status", "")).startswith("skipped")),
        "errored_pages": sum(
            1
            for p in report["pages"]
            if p.get("status") in {"pred_load_error", "invalid_gt_page_index"}
        ),
        "overall_score": None if not page_scores else sum(page_scores) / len(page_scores),
    }

    report_path = config.get("output", {}).get("report_path", "benchmark_output/report.json")
    dump_json(report_path, report)

    print("\nSummary")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
