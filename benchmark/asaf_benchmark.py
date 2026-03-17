import json
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = "/Users/delmedigo/Dev/FineTree/benchmark/benchmark_config.yaml"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).replace("״", '"').replace("(*)", "").strip()


def to_comparable_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def char_similarity(a, b) -> float:
    a = normalize_text(to_comparable_string(a))
    b = normalize_text(to_comparable_string(b))
    return SequenceMatcher(None, a, b).ratio()


def mean_or_none(values):
    if not values:
        return None
    return sum(values) / len(values)


def parse_pages_to_skip(value) -> set[int]:
    if value is None:
        return set()

    if isinstance(value, int):
        return {value}

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return {int(part) for part in parts if part}

    if isinstance(value, list):
        pages = set()
        for item in value:
            if isinstance(item, int):
                pages.add(item)
            elif isinstance(item, str):
                item = item.strip()
                if item:
                    pages.add(int(item))
            else:
                raise ValueError(f"Unsupported pages_to_skip entry: {item!r}")
        return pages

    raise ValueError(f"Unsupported pages_to_skip value: {value!r}")


def get_meta_container(page: dict) -> dict:
    meta = page.get("meta")
    if isinstance(meta, dict):
        return meta
    return {}


def extract_field_value(item: dict, field_name: str, method: str):
    value = item.get(field_name)

    if method == "path_exact":
        if isinstance(value, list) and len(value) > 0:
            return value[-1]
        return value

    return value


def compare_values(pred_value, gt_value, method: str) -> float:
    if method == "char_similarity":
        return float(char_similarity(pred_value, gt_value))

    if method == "string_normalized":
        return float(
            normalize_text(to_comparable_string(pred_value))
            == normalize_text(to_comparable_string(gt_value))
        )

    if method == "path_exact":
        return float(
            normalize_text(to_comparable_string(pred_value))
            == normalize_text(to_comparable_string(gt_value))
        )

    if method == "equality":
        if isinstance(pred_value, str) or isinstance(gt_value, str):
            return float(normalize_text(pred_value) == normalize_text(gt_value))
        return float(pred_value == gt_value)

    raise ValueError(f"Unsupported method: {method}")


def compare_meta_fields(pred_page: dict, gt_page: dict, meta_fields_cfg: dict) -> pd.DataFrame:
    pred_meta = get_meta_container(pred_page)
    gt_meta = get_meta_container(gt_page)

    rows = []
    for field_name, field_cfg in meta_fields_cfg.items():
        method = field_cfg["method"]

        pred_value = pred_meta.get(field_name)
        gt_value = gt_meta.get(field_name)

        score = compare_values(pred_value=pred_value, gt_value=gt_value, method=method)

        rows.append(
            {
                "field": field_name,
                "method": method,
                "pred_value": pred_value,
                "gt_value": gt_value,
                "score": float(score),
            }
        )

    return pd.DataFrame(rows)


def compare_fact_fields(
    pred_page: dict,
    gt_page: dict,
    fact_fields_cfg: dict,
    facts_key: str,
) -> pd.DataFrame:
    pred_facts = pred_page.get(facts_key, [])
    gt_facts = gt_page.get(facts_key, [])

    rows = []

    for fact_idx, (pred_fact, gt_fact) in enumerate(zip(pred_facts, gt_facts), start=1):
        pred_fact_num = pred_fact.get("fact_num", fact_idx)
        gt_fact_num = gt_fact.get("fact_num", fact_idx)

        for field_name, field_cfg in fact_fields_cfg.items():
            method = field_cfg["method"]

            pred_value = extract_field_value(pred_fact, field_name, method)
            gt_value = extract_field_value(gt_fact, field_name, method)

            score = compare_values(pred_value=pred_value, gt_value=gt_value, method=method)

            rows.append(
                {
                    "fact_idx": fact_idx,
                    "pred_fact_num": pred_fact_num,
                    "gt_fact_num": gt_fact_num,
                    "field": field_name,
                    "method": method,
                    "pred_value": pred_value,
                    "gt_value": gt_value,
                    "score": float(score),
                }
            )

    return pd.DataFrame(rows)


def update_field_scores(store: dict, df: pd.DataFrame):
    if df.empty:
        return

    for field_name, group in df.groupby("field"):
        scores = [float(x) for x in group["score"].tolist()]
        store.setdefault(field_name, []).extend(scores)


def main():
    config = load_yaml(CONFIG_PATH)

    dataset_cfg = config["dataset"]
    evaluation_cfg = config["evaluation"]
    output_cfg = config["output"]
    meta_fields_cfg = config.get("meta_fields", {})
    fact_fields_cfg = config.get("fact_fields", {})

    gt_path = dataset_cfg["gt_path"]
    start_idx = dataset_cfg["start_idx"]
    end_idx = dataset_cfg["end_idx"]  # exclusive
    pages_to_skip = parse_pages_to_skip(dataset_cfg.get("pages_to_skip"))
    pred_pattern = dataset_cfg["pred_pattern"]

    facts_key = evaluation_cfg.get("facts_key", "facts")
    skip_0 = evaluation_cfg.get("skip_0", True)
    skip_mismatch = evaluation_cfg.get("skip_mismatch", True)

    report_path = Path(output_cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("CONFIG_PATH =", CONFIG_PATH)
    print("gt_path =", gt_path)
    print("start_idx =", start_idx)
    print("end_idx =", end_idx)
    print("pages_to_skip =", sorted(pages_to_skip))
    print("facts_key =", facts_key)
    print("report_path =", report_path)
    print()

    gt_data = load_json(gt_path)

    print("GT page fact counts in active run:")
    for i in range(start_idx, end_idx):
        if i in pages_to_skip:
            print(f"page {i}: skipped by config")
            continue
        page = gt_data["pages"][i - 1]
        print(f"page {i}: {len(page.get(facts_key, []))}")
    print()

    page_scores = []
    meta_field_scores = {field_name: [] for field_name in meta_fields_cfg.keys()}
    fact_field_scores = {field_name: [] for field_name in fact_fields_cfg.keys()}
    report_pages = []

    for idx in range(start_idx, end_idx):
        pred_path = pred_pattern.format(idx=idx)

        try:
            if idx in pages_to_skip:
                print(f"[idx={idx}] skipped by config")
                report_pages.append(
                    {
                        "idx": idx,
                        "pred_path": pred_path,
                        "status": "skipped_config",
                    }
                )
                continue

            gt_page = gt_data["pages"][idx - 1]
            pred_page = load_json(pred_path)

            for i, fact in enumerate(pred_page.get(facts_key, []), start=1):
                fact["fact_num"] = i

            pred_facts = pred_page.get(facts_key, [])
            gt_facts = gt_page.get(facts_key, [])

            print(f"[idx={idx}] pred_path={pred_path}")
            print(f"[idx={idx}] gt_facts_count={len(gt_facts)} pred_facts_count={len(pred_facts)}")

            meta_df = compare_meta_fields(
                pred_page=pred_page,
                gt_page=gt_page,
                meta_fields_cfg=meta_fields_cfg,
            )
            update_field_scores(meta_field_scores, meta_df)

            meta_scores = [float(x) for x in meta_df["score"].tolist()] if not meta_df.empty else []
            meta_score = mean_or_none(meta_scores)

            facts_df = pd.DataFrame()
            facts_score = None
            facts_status = "not_run"

            if len(gt_facts) == 0 and skip_0:
                print(f"--- comparing {idx} idx ---")
                print("gt facts = 0 -> skipping FACT comparison, evaluating META only")
                print("META")
                print(meta_df)
                print("---------------------\n")

                facts_status = "skipped_no_gt_facts"

            elif len(pred_facts) == 0 and skip_0:
                print(f"--- comparing {idx} idx ---")
                print("pred facts = 0 -> skipping FACT comparison, evaluating META only")
                print("META")
                print(meta_df)
                print("---------------------\n")

                facts_status = "skipped_no_pred_facts"

            elif len(pred_facts) != len(gt_facts):
                print(f"--- comparing {idx} idx ---")
                print(f"length mismatch -> pred={len(pred_facts)}, gt={len(gt_facts)}")

                if skip_mismatch:
                    print("skipping FACT comparison, evaluating META only")
                    print("META")
                    print(meta_df)
                    print("---------------------\n")
                    facts_status = "skipped_length_mismatch"
                else:
                    print("FACT score forced to 0 because of length mismatch")
                    print("META")
                    print(meta_df)
                    print("---------------------\n")
                    facts_status = "length_mismatch_scored_zero"
                    facts_score = 0.0

            else:
                facts_df = compare_fact_fields(
                    pred_page=pred_page,
                    gt_page=gt_page,
                    fact_fields_cfg=fact_fields_cfg,
                    facts_key=facts_key,
                )
                update_field_scores(fact_field_scores, facts_df)

                fact_scores = [float(x) for x in facts_df["score"].tolist()] if not facts_df.empty else []
                facts_score = mean_or_none(fact_scores)
                facts_status = "evaluated"

                print(f"--- comparing {idx} idx ---")
                print("META")
                print(meta_df)
                print("\nFACTS")
                print(facts_df.head(20))
                print("---------------------\n")

            combined_scores = []
            if meta_score is not None:
                combined_scores.append(float(meta_score))
            if facts_score is not None:
                combined_scores.append(float(facts_score))

            page_score = mean_or_none(combined_scores)

            if page_score is not None:
                page_scores.append(float(page_score))

            report_pages.append(
                {
                    "idx": idx,
                    "pred_path": pred_path,
                    "status": "evaluated" if facts_status == "evaluated" else "evaluated_meta_only",
                    "facts_status": facts_status,
                    "pred_facts_count": len(pred_facts),
                    "gt_facts_count": len(gt_facts),
                    "meta_score": meta_score,
                    "facts_score": facts_score,
                    "page_score": page_score,
                    "meta_results": meta_df.to_dict(orient="records"),
                    "fact_results": facts_df.to_dict(orient="records") if not facts_df.empty else [],
                }
            )

        except Exception as e:
            print("error", e, "with", idx)
            report_pages.append(
                {
                    "idx": idx,
                    "pred_path": pred_path,
                    "status": "error",
                    "error": str(e),
                }
            )

    report = {
        "config_path": str(CONFIG_PATH),
        "summary": {
            "pages_skipped_config": len([p for p in report_pages if p["status"] == "skipped_config"]),
            "pages_evaluated": len([p for p in report_pages if p["status"] in {"evaluated", "evaluated_meta_only"}]),
            "pages_fully_evaluated": len([p for p in report_pages if p["status"] == "evaluated"]),
            "pages_meta_only": len([p for p in report_pages if p["status"] == "evaluated_meta_only"]),
            "overall_score": mean_or_none(page_scores),
            "meta_field_scores": {
                field_name: mean_or_none(scores)
                for field_name, scores in meta_field_scores.items()
            },
            "fact_field_scores": {
                field_name: mean_or_none(scores)
                for field_name, scores in fact_field_scores.items()
            },
        },
        "pages": report_pages,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("page_scores =", page_scores)
    if page_scores:
        print("Final overall score:", mean_or_none(page_scores))
    else:
        print("No measurements collected.")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
