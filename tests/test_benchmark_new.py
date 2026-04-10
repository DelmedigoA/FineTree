from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

from benchmark_new.cli import main
from benchmark_new.eval import evaluate_single_page
from benchmark_new.evaluation_specs import get_evaluator_spec, get_facts_field_spec, load_evaluation_specs, load_evaluation_specs_from_path
from benchmark_new.interactive import dataset_option_label, execute_interactive_evaluation, list_native_inference_runs, run_interactive_benchmark
from benchmark_new.models import DocumentResult, NativeRunInfo, RunResult
from benchmark_new.io.datasets import list_dataset_versions, load_dataset_selection
from benchmark_new.io.run_adapters import load_predictions_from_run_dir
from benchmark_new.reports import (
    build_path_comparison_report,
    write_full_metrics_csv,
    write_mistakes_json,
    write_mistakes_values_json,
    write_values_mistakes_json,
)
from benchmark_new.spec import normalize_fact_value, parse_numeric_value


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def _write_annotation(data_root: Path, *, doc_id: str, page_payload: dict[str, object]) -> None:
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"{doc_id}.json").write_text(
        json.dumps({"pages": [page_payload]}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_native_run(
    run_dir: Path,
    *,
    doc_id: str,
    parsed_page: dict[str, object],
    dataset_version_id: str,
    dataset_name: str = "dataset",
    split: str = "val",
    model: str = "model-a",
    started_at: float = 1.0,
) -> None:
    pages_dir = run_dir / "documents" / doc_id / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "provider": "native-test",
                "model": model,
                "dataset_version_id": dataset_version_id,
                "dataset_name": dataset_name,
                "split": split,
                "started_at": started_at,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (pages_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "page_index": 1,
                "page_name": "page_0001.png",
                "assistant_text": json.dumps(parsed_page, ensure_ascii=False),
                "parsed_page": parsed_page,
                "received_tokens": 10,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_evaluate_single_page_exact_match() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [
            {
                "value": "10",
                "fact_num": 1,
                "period_type": "instant",
                "period_start": "2024-01-01",
                "period_end": "2024-01-01",
                "path": ["Assets", "Cash"],
                "currency": "ILS",
                "scale": 1,
                "value_type": "amount",
                "value_context": "tabular",
            }
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, gt_page)
    assert result.meta_score == 1.0
    assert result.facts_score == 1.0
    assert result.page_score == 1.0


def test_soft_meta_quote_normalization_scores_as_exact_match() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": '"המעין" קרן מלגות למחוננים (ע"ר)',
            "page_num": "1",
            "page_type": "contents",
            "statement_type": None,
            "title": "דוח שנתי",
        },
        "facts": [],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": '״המעין״ קרן מלגות למחוננים (ע״ר)',
            "page_num": "1",
            "page_type": "contents",
            "statement_type": None,
            "title": "דוח שנתי",
        },
        "facts": [],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.meta_result["entity_name"].score == 1.0
    assert result.meta_result["entity_name"].is_exact_match is True


def test_soft_meta_lowercase_normalization_scores_as_exact_match() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme Foundation",
            "page_num": "1",
            "page_type": "contents",
            "statement_type": None,
            "title": "Annual Report",
        },
        "facts": [],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "acme foundation",
            "page_num": "1",
            "page_type": "contents",
            "statement_type": None,
            "title": "annual report",
        },
        "facts": [],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.meta_result["entity_name"].score == 1.0
    assert result.meta_result["title"].score == 1.0


def test_evaluate_single_page_empty_facts_match_scores_as_pass() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.gt_fact_count == 0
    assert result.pred_fact_count == 0
    assert result.facts_status == "empty_match"
    assert result.facts_score == 1.0
    assert result.page_score == 1.0


def test_evaluate_single_page_empty_gt_with_false_positive_uses_count_scaled_penalty() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": []}
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"value": "10", "fact_num": 1},
            {"value": "20", "fact_num": 2},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.gt_fact_count == 0
    assert result.pred_fact_count == 2
    assert result.facts_status == "false_positive_on_empty_gt"
    assert result.facts_score == 1.0 / 3.0


def test_numeric_and_date_helpers_do_not_penalize_when_no_parseable_pairs() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {
                "value": "unchanged text",
                "fact_num": 1,
                "period_type": "instant",
                "period_start": "not-a-date",
                "period_end": "still-not-a-date",
                "path": ["Assets"],
                "currency": "ILS",
                "scale": 1,
                "value_type": "amount",
                "value_context": "tabular",
            }
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, gt_page)
    assert result.facts_result["value"].exact_metrics.f1 == 1.0
    assert result.facts_result["value"].score == 1.0
    assert result.facts_result["value"].details["numeric_closeness_score"] is None
    assert result.facts_result["value"].details["helper_pair_count"] == 0
    assert result.facts_result["period_start"].exact_metrics.f1 == 1.0
    assert result.facts_result["period_start"].score == 1.0
    assert result.facts_result["period_start"].details["date_diff_score"] is None
    assert result.facts_result["period_start"].details["helper_pair_count"] == 0


def test_value_exact_match_ignores_thousands_separator_differences() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "value": "745384", "path": ["רכוש", "רכוש שוטף", "מזומנים ושווי מזומנים"]},
            {"fact_num": 2, "value": "851625", "path": ["רכוש", "רכוש שוטף", "מזומנים ושווי מזומנים"]},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "value": "745,384", "path": ["רכוש", "רכוש שוטף", "מזומנים ושווי מזומנים"]},
            {"fact_num": 2, "value": "851,625", "path": ["רכוש", "רכוש שוטף", "מזומנים ושווי מזומנים"]},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.facts_result["value"].exact_metrics.f1 == 1.0
    assert result.facts_result["value"].score == 1.0
    assert result.row_diff_diagnostics[0]["op"] == "equal"


def test_value_normalization_handles_negative_parentheses_and_footnote_tokens() -> None:
    assert normalize_fact_value("value", "- 1,522") == "(1522)"
    assert normalize_fact_value("value", "-1,522.50") == "(1522.50)"
    assert normalize_fact_value("value", "(*) 39,396,389") == "39396389"
    assert normalize_fact_value("value", "(*)-") == "0"
    assert parse_numeric_value("(1522)") == -1522.0


def test_value_alignment_and_accuracy_are_scored_separately() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "10"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "20"},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "10"},
            {"fact_num": 3, "path": ["Equity"], "value": "999"},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    value_result = result.facts_result["value"]
    assert value_result.exact_metrics.precision == 0.5
    assert value_result.exact_metrics.recall == 0.5
    assert value_result.exact_metrics.f1 == 0.5
    assert value_result.details["value_accuracy"] == 1.0
    assert value_result.details["value_accuracy_scope"] == "matched_pairs"
    assert value_result.details["matched_pair_count"] == 1
    assert value_result.score == 0.75


def test_path_threshold_checker_exact_match_passes() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert path_result.score == 1.0
    assert path_result.exact_metrics.accuracy == 1.0
    assert path_result.details["path_method_scores"]["path_all_levels_threshold"] == 1.0
    assert path_result.details["path_method_scores"]["path_last_leaf_threshold"] == 1.0
    assert path_result.details["path_method_scores"]["path_soft_overlap"] == 1.0


def test_path_threshold_checker_small_spelling_difference_passes() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["נכסים", "לקוחות"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["נכסי", "לקוחות"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert path_result.score == 1.0
    assert path_result.exact_metrics.accuracy == 1.0
    assert path_result.details["path_method_scores"]["path_all_levels_threshold"] == 1.0
    assert path_result.details["path_method_scores"]["path_last_leaf_threshold"] == 1.0
    assert path_result.details["path_method_scores"]["path_soft_overlap"] == 1.0


def test_path_threshold_checker_element_below_threshold_fails() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Equity"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert path_result.score == 0.0
    assert path_result.exact_metrics.accuracy == 0.0
    assert path_result.details["path_method_scores"]["path_all_levels_threshold"] == 0.0
    assert path_result.details["path_method_scores"]["path_last_leaf_threshold"] == 0.0
    assert path_result.details["path_method_scores"]["path_soft_overlap"] == 0.0


def test_path_soft_overlap_missing_parent_node_gets_partial_credit() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Current", "Cash"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert round(path_result.score, 6) == round((0.0 + 1.0 + 0.8) / 3.0, 6)
    assert path_result.details["path_method_scores"]["path_all_levels_threshold"] == 0.0
    assert path_result.details["path_method_scores"]["path_last_leaf_threshold"] == 1.0
    assert round(path_result.details["path_method_scores"]["path_soft_overlap"], 6) == round(0.8, 6)


def test_path_soft_overlap_extra_node_in_prediction_gets_partial_credit() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Current", "Cash"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert round(path_result.score, 6) == round((0.0 + 1.0 + 0.8) / 3.0, 6)
    assert path_result.exact_metrics.accuracy == 0.0
    assert path_result.details["path_method_scores"]["path_all_levels_threshold"] == 0.0
    assert path_result.details["path_method_scores"]["path_last_leaf_threshold"] == 1.0
    assert round(path_result.details["path_method_scores"]["path_soft_overlap"], 6) == round(0.8, 6)


def test_path_soft_overlap_completely_different_path_fails() -> None:
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Liabilities", "Debt"]}]}
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert path_result.details["path_method_scores"]["path_soft_overlap"] == 0.0


def test_path_threshold_checker_matches_rows_on_value() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"value": "10", "path": ["Assets", "Cash"]},
            {"value": "20", "path": ["Liabilities", "Debt"]},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"value": "20", "path": ["Liabilities", "Debt"]},
            {"value": "10", "path": ["Assets", "Cash"]},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    path_result = result.facts_result["path"]
    assert path_result.details["matching_on"] == "value"
    assert path_result.exact_metrics.accuracy == 0.5
    assert round(path_result.score, 6) == round((0.5 + 0.5 + 1.0) / 3.0, 6)


def test_non_value_fact_fields_match_rows_on_value_lcs() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"value": "10", "currency": "ILS", "comment_ref": "ביאור א"},
            {"value": "20", "currency": "USD", "comment_ref": "ביאור ב"},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"value": "20", "currency": "USD", "comment_ref": "ביאור ב"},
            {"value": "10", "currency": "ILS", "comment_ref": "ביאור א"},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    assert result.facts_result["currency"].exact_metrics.accuracy == 0.5
    assert result.facts_result["currency"].exact_metrics.matches == 1
    assert result.facts_result["comment_ref"].exact_metrics.accuracy == 0.5
    assert result.facts_result["comment_ref"].details["helper_pair_count"] == 1


def test_facts_summary_aggregates_at_fact_level_not_page_level(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "10"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "20"},
            {"fact_num": 3, "path": ["Equity"], "value": "30"},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "10"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "999"},
        ],
    }
    _write_annotation(data_root, doc_id="doc1", page_payload=gt_page)
    run_dir = tmp_path / "outputs" / "run1"
    _write_native_run(run_dir, doc_id="doc1", parsed_page=pred_page, dataset_version_id="dataset-1")
    from benchmark_new.eval import summarize_facts_run
    from benchmark_new.io.run_adapters import load_predictions_from_run_dir

    bundle = load_predictions_from_run_dir(run_dir, data_root=data_root)
    run_result = evaluate_predictions_bundle(bundle, data_root=data_root)
    summary = summarize_facts_run(run_result, run_dir=run_dir, manifest=bundle.get("manifest"))
    channels = {(channel.channel, channel.metric): channel for channel in summary.channels}
    assert summary.fact_count == 3
    assert channels[("alignment", "precision")].score == 0.5
    assert channels[("alignment", "recall")].score == 1.0 / 3.0
    assert channels[("alignment", "f1")].score == 0.4
    assert channels[("value", "accuracy")].score == 1.0
    assert summary.overall_score == 0.7


def test_full_metrics_csv_contains_fact_previews_and_status_columns(tmp_path: Path) -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [{"value": "10", "fact_num": 1, "path": ["Assets", "Cash"]}],
    }
    page = evaluate_single_page("acme", 1, gt_page, gt_page)
    run_result = RunResult(
        run_score=page.page_score,
        meta_score=page.meta_score,
        facts_score=page.facts_score,
        documents=[
            DocumentResult(
                doc_id="acme",
                meta_score=page.meta_score,
                facts_score=page.facts_score,
                document_score=page.page_score,
                pages=[page],
            )
        ],
    )
    output_path = tmp_path / "full_metrics.csv"
    write_full_metrics_csv(
        output_path,
        run_result=run_result,
        provider="finetree_vllm",
        dataset_version_id="dataset-id",
        dataset_name="dataset",
        split="val",
    )
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    value_row = next(row for row in rows if row["section"] == "facts" and row["field"] == "value")
    assert value_row["gt_value_preview"] == "10"
    assert value_row["pred_value_preview"] == "10"
    assert value_row["gt_fact_count"] == "1"
    assert value_row["pred_fact_count"] == "1"
    assert value_row["facts_status"] == "standard"
    assert "helper_pair_count" in value_row


def test_sparse_optional_field_preview_shows_populated_values_only() -> None:
    gt_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "comment_ref": None},
            {"fact_num": 2, "comment_ref": None},
            {"fact_num": 3, "comment_ref": "ביאור א"},
            {"fact_num": 4, "comment_ref": None},
        ],
    }
    pred_page = {
        "image": "page_0001.png",
        "meta": {},
        "facts": [
            {"fact_num": 1, "comment_ref": None},
            {"fact_num": 2, "comment_ref": None},
            {"fact_num": 3, "comment_ref": "ביאור א"},
            {"fact_num": 4, "comment_ref": "ביאור ב"},
        ],
    }
    result = evaluate_single_page("acme", 1, gt_page, pred_page)
    details = result.facts_result["comment_ref"].details
    assert details["gt_preview"] == "ביאור א"
    assert details["pred_preview"] == "ביאור א | ביאור ב"
    assert details["gt_non_null_count"] == 1
    assert details["pred_non_null_count"] == 2
    assert details["gt_null_count"] == 3
    assert details["pred_null_count"] == 2


def test_write_mistakes_json_reports_element_level_mismatches(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "doc1.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {
                            "entity_name": "Acme",
                            "page_num": "1",
                            "page_type": "title",
                            "statement_type": None,
                            "title": "Annual",
                        },
                        "facts": [
                            {"fact_num": 1, "value": "10", "comment_ref": None},
                            {"fact_num": 2, "value": "20", "comment_ref": "ביאור א"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pred_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [
            {"fact_num": 1, "value": "10", "comment_ref": None},
            {"fact_num": 2, "value": "30", "comment_ref": None},
            {"fact_num": 3, "value": "40", "comment_ref": "ביאור ב"},
        ],
    }
    page = evaluate_single_page(
        "doc1",
        1,
        json.loads((annotations_dir / "doc1.json").read_text(encoding="utf-8"))["pages"][0],
        pred_page,
    )
    run_result = RunResult(
        run_score=page.page_score,
        meta_score=page.meta_score,
        facts_score=page.facts_score,
        documents=[
            DocumentResult(
                doc_id="doc1",
                meta_score=page.meta_score,
                facts_score=page.facts_score,
                document_score=page.page_score,
                pages=[page],
            )
        ],
    )
    bundle = {
        "documents": {
            "doc1": {
                "doc_id": "doc1",
                "pages": [
                    {
                        "page_index": 1,
                        "page_name": "page_0001.png",
                        "parsed_page": pred_page,
                    }
                ],
            }
        }
    }
    output_path = tmp_path / "mistakes.json"
    write_mistakes_json(output_path, run_result=run_result, bundle=bundle, data_root=data_root)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    page_payload = payload["documents"][0]["pages"][0]
    value_issue = next(item for item in page_payload["fact_mistakes"] if item["field"] == "value")
    comment_issue = next(item for item in page_payload["fact_mistakes"] if item["field"] == "comment_ref")
    assert value_issue["elements"][0]["element_index"] == 2
    assert value_issue["elements"][0]["gt_value"] == "20"
    assert value_issue["elements"][0]["pred_value"] == "30"
    assert comment_issue["elements"][0]["element_index"] == 2
    assert comment_issue["elements"][0]["gt_value"] == "ביאור א"
    assert comment_issue["elements"][0]["pred_value"] is None


def test_write_mistakes_values_json_reports_page_value_lists_and_matches(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "doc1.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {"entity_name": "Acme"},
                        "facts": [
                            {"fact_num": 1, "path": ["Assets"], "value": "1,000"},
                            {"fact_num": 2, "path": ["Liabilities"], "value": "20"},
                            {"fact_num": 3, "path": ["Equity"], "value": "30"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pred_page = {
        "image": "page_0001.png",
        "meta": {"entity_name": "Acme"},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "1000"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "25"},
            {"fact_num": 4, "path": ["Other"], "value": "40"},
        ],
    }
    gt_page = json.loads((annotations_dir / "doc1.json").read_text(encoding="utf-8"))["pages"][0]
    page = evaluate_single_page("doc1", 1, gt_page, pred_page)
    run_result = RunResult(
        run_score=page.page_score,
        meta_score=page.meta_score,
        facts_score=page.facts_score,
        documents=[
            DocumentResult(
                doc_id="doc1",
                meta_score=page.meta_score,
                facts_score=page.facts_score,
                document_score=page.page_score,
                pages=[page],
            )
        ],
    )
    bundle = {
        "documents": {
            "doc1": {
                "doc_id": "doc1",
                "pages": [
                    {
                        "page_index": 1,
                        "page_name": "page_0001.png",
                        "parsed_page": pred_page,
                    }
                ],
            }
        }
    }
    output_path = tmp_path / "mistakes_values.json"
    write_mistakes_values_json(output_path, run_result=run_result, bundle=bundle, data_root=data_root)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    page_payload = payload["documents"][0]["pages"][0]
    assert [item["value"] for item in page_payload["gt_values"]] == ["1000", "20", "30"]
    assert [item["value"] for item in page_payload["pred_values"]] == ["1000", "25", "40"]
    assert page_payload["value_matching"]["matcher"] == "lcs"
    assert page_payload["value_matching"]["alignment_on"] == ["value"]
    assert page_payload["value_matching"]["matched_pair_count"] == 1
    assert page_payload["value_matching"]["mismatched_pair_count"] == 0
    assert page_payload["value_matching"]["unmatched_gt_count"] == 2
    assert page_payload["value_matching"]["unmatched_pred_count"] == 2
    assert page_payload["value_matching"]["matched_pairs"][0]["value_match"] is True
    assert page_payload["value_matching"]["mismatched_pairs"] == []
    assert [item["value"] for item in page_payload["value_matching"]["unmatched_gt"]] == ["20", "30"]
    assert [item["value"] for item in page_payload["value_matching"]["unmatched_pred"]] == ["25", "40"]


def test_write_values_mistakes_json_reports_simple_flat_value_mistakes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    gt_page = {
        "image": "page_0001.png",
        "meta": {"entity_name": "Acme"},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "1000"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "20"},
            {"fact_num": 3, "path": ["Equity"], "value": "30"},
        ],
    }
    (annotations_dir / "doc1.json").write_text(json.dumps({"pages": [gt_page]}, ensure_ascii=False), encoding="utf-8")
    pred_page = {
        "image": "page_0001.png",
        "meta": {"entity_name": "Acme"},
        "facts": [
            {"fact_num": 1, "path": ["Assets"], "value": "1000"},
            {"fact_num": 2, "path": ["Liabilities"], "value": "25"},
            {"fact_num": 4, "path": ["Other"], "value": "40"},
        ],
    }
    page = evaluate_single_page("doc1", 1, gt_page, pred_page)
    run_result = RunResult(
        run_score=page.page_score,
        meta_score=page.meta_score,
        facts_score=page.facts_score,
        documents=[
            DocumentResult(
                doc_id="doc1",
                meta_score=page.meta_score,
                facts_score=page.facts_score,
                document_score=page.page_score,
                pages=[page],
            )
        ],
    )
    bundle = {
        "documents": {
            "doc1": {
                "doc_id": "doc1",
                "pages": [
                    {
                        "page_index": 1,
                        "page_name": "page_0001.png",
                        "parsed_page": pred_page,
                    }
                ],
            }
        }
    }
    output_path = tmp_path / "values_mistakes.json"
    write_values_mistakes_json(output_path, run_result=run_result, bundle=bundle, data_root=data_root)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["mistake_count"] == 4
    assert payload["items"][0]["doc_id"] == "doc1"
    assert payload["items"][0]["page_index"] == 1
    assert {item["kind"] for item in payload["items"]} == {"unmatched_gt", "unmatched_pred"}
    assert [item["gt"]["value"] for item in payload["items"] if item["kind"] == "unmatched_gt"] == ["20", "30"]
    assert [item["pred"]["value"] for item in payload["items"] if item["kind"] == "unmatched_pred"] == ["25", "40"]


def test_build_path_comparison_report_groups_examples_by_error_type(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    gt_pages = {
        "pages": [
            {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Current", "Cash"]}]},
            {"image": "page_0002.png", "meta": {}, "facts": [{"value": "20", "path": ["Assets", "Cash"]}]},
            {"image": "page_0003.png", "meta": {}, "facts": [{"value": "30", "path": ["נכסים", "לקוחות"]}]},
            {"image": "page_0004.png", "meta": {}, "facts": [{"value": "40", "path": ["Assets", "Cash"]}]},
        ]
    }
    (annotations_dir / "doc1.json").write_text(json.dumps(gt_pages, ensure_ascii=False), encoding="utf-8")
    pred_pages = [
        {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["Assets", "Cash"]}]},
        {"image": "page_0002.png", "meta": {}, "facts": [{"value": "20", "path": ["Assets", "Current", "Cash"]}]},
        {"image": "page_0003.png", "meta": {}, "facts": [{"value": "30", "path": ["נכסי", "לקוחות"]}]},
        {"image": "page_0004.png", "meta": {}, "facts": [{"value": "40", "path": ["Liabilities", "Debt"]}]},
    ]
    pages = [evaluate_single_page("doc1", index, gt_pages["pages"][index - 1], pred_pages[index - 1]) for index in range(1, 5)]
    run_result = RunResult(
        run_score=sum(page.page_score for page in pages) / len(pages),
        meta_score=sum(page.meta_score for page in pages) / len(pages),
        facts_score=sum(page.facts_score for page in pages) / len(pages),
        documents=[DocumentResult(doc_id="doc1", meta_score=0.0, facts_score=0.0, document_score=0.0, pages=pages)],
    )
    bundle = {
        "documents": {
            "doc1": {
                "doc_id": "doc1",
                "pages": [
                    {"page_index": index, "page_name": f"page_{index:04d}.png", "parsed_page": pred_pages[index - 1]}
                    for index in range(1, 5)
                ],
            }
        }
    }
    report = build_path_comparison_report(run_result=run_result, bundle=bundle, data_root=data_root)
    counts = report["counts"]
    assert counts["missing_ancestor"] == 1
    assert counts["extra_ancestor"] == 1
    assert counts["wrong_ancestor"] == 1
    assert counts["wrong_leaf"] == 1
    assert report["groups"][0]["count"] >= report["groups"][-1]["count"]
    assert report["representative_examples"]


def test_build_path_comparison_report_classifies_same_leaf_different_parent_as_wrong_ancestor(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    gt_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["רכוש", 'סה"כ']}]}
    pred_page = {"image": "page_0001.png", "meta": {}, "facts": [{"value": "10", "path": ["נכסים", 'סה"כ']}]}
    (annotations_dir / "doc1.json").write_text(json.dumps({"pages": [gt_page]}, ensure_ascii=False), encoding="utf-8")
    page = evaluate_single_page("doc1", 1, gt_page, pred_page)
    run_result = RunResult(
        run_score=page.page_score,
        meta_score=page.meta_score,
        facts_score=page.facts_score,
        documents=[DocumentResult(doc_id="doc1", meta_score=page.meta_score, facts_score=page.facts_score, document_score=page.page_score, pages=[page])],
    )
    bundle = {
        "documents": {
            "doc1": {
                "doc_id": "doc1",
                "pages": [
                    {"page_index": 1, "page_name": "page_0001.png", "parsed_page": pred_page}
                ],
            }
        }
    }
    report = build_path_comparison_report(run_result=run_result, bundle=bundle, data_root=data_root)
    assert report["counts"]["wrong_ancestor"] == 1
    assert report["counts"]["wrong_leaf"] == 0
    example = report["groups"][0]["examples"][0]
    assert example["gt_path"] == ["רכוש", 'סה"כ']
    assert example["pred_path"] == ["נכסים", 'סה"כ']


def test_list_dataset_versions_reads_workspace_dataset_file() -> None:
    versions = list_dataset_versions(data_root=Path("/Users/delmedigo/Dev/FineTree/data"))
    assert versions
    assert any(version.version_id == "b9172927-e1c9-40ca-81a5-963f0440ce07" for version in versions)


def test_load_dataset_selection_returns_val_documents() -> None:
    selection = load_dataset_selection(
        "b9172927-e1c9-40ca-81a5-963f0440ce07",
        split="val",
        data_root=Path("/Users/delmedigo/Dev/FineTree/data"),
    )
    assert selection.documents
    assert selection.split == "val"
    assert all(document.pages for document in selection.documents)


def test_load_predictions_from_vllm_parallel_run() -> None:
    run_dir = Path(
        "/Users/delmedigo/Dev/FineTree/vllm_api_tests/outputs_parallel/20260403_202744/דוח כספי 2020 - -חוף הגליל- אגודה לטפול בילד ובמשפחה _ע-ר_"
    )
    bundle = load_predictions_from_run_dir(run_dir, data_root=Path("/Users/delmedigo/Dev/FineTree/data"))
    assert bundle["kind"] == "vllm_results"
    assert bundle["documents"]
    first_document = next(iter(bundle["documents"].values()))
    assert first_document["pages"]


def test_eval_cli_writes_reports_for_native_run(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    annotations_dir = data_root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "doc1.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "image": "page_0001.png",
                        "meta": {
                            "entity_name": "Acme",
                            "page_num": "1",
                            "page_type": "title",
                            "statement_type": None,
                            "title": "Annual",
                        },
                        "facts": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    pages_dir = run_dir / "documents" / "doc1" / "pages"
    pages_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"provider": "native-test"}), encoding="utf-8")
    (pages_dir / "page_0001.json").write_text(
        json.dumps(
            {
                "page_index": 1,
                "page_name": "page_0001.png",
                "assistant_text": json.dumps(
                    {
                        "meta": {
                            "entity_name": "Acme",
                            "page_num": "1",
                            "page_type": "title",
                            "statement_type": None,
                            "title": "Annual",
                        },
                        "facts": [],
                    },
                    ensure_ascii=False,
                ),
                "parsed_page": {
                    "image": "page_0001.png",
                    "meta": {
                        "entity_name": "Acme",
                        "page_num": "1",
                        "page_type": "title",
                        "statement_type": None,
                        "title": "Annual",
                    },
                    "facts": [],
                },
                "received_tokens": 10,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    exit_code = main(["eval", "--run-dir", str(run_dir), "--data-root", str(data_root)])
    assert exit_code == 0
    assert (run_dir / "evaluation" / "run_metrics.json").is_file()
    assert (run_dir / "evaluation" / "summary.csv").is_file()
    assert (run_dir / "evaluation" / "full_metrics.csv").is_file()
    assert (run_dir / "evaluation" / "mistakes.json").is_file()
    assert (run_dir / "evaluation" / "mistakes_values.json").is_file()
    assert (run_dir / "evaluation" / "values_mistakes.json").is_file()
    assert (run_dir / "evaluation" / "path_comparison_report.json").is_file()
    with (run_dir / "evaluation" / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "meta__entity_name__score" in rows[0]
    assert "facts__value__f1" in rows[0]
    assert "facts__path__score" in rows[0]
    mistakes = json.loads((run_dir / "evaluation" / "mistakes.json").read_text(encoding="utf-8"))
    assert mistakes["documents"] == []
    values_report = json.loads((run_dir / "evaluation" / "mistakes_values.json").read_text(encoding="utf-8"))
    assert values_report["page_count"] == 1
    assert values_report["documents"][0]["pages"][0]["gt_values"] == []
    simple_values_report = json.loads((run_dir / "evaluation" / "values_mistakes.json").read_text(encoding="utf-8"))
    assert simple_values_report["mistake_count"] == 0
    comparison_report = json.loads((run_dir / "evaluation" / "path_comparison_report.json").read_text(encoding="utf-8"))
    assert "counts" in comparison_report


def test_load_evaluation_specs_reads_page_meta_yaml() -> None:
    specs = load_evaluation_specs()
    page_meta = get_evaluator_spec("page_meta")
    facts = get_evaluator_spec("facts")
    facts_value = get_facts_field_spec("value")
    facts_path = get_facts_field_spec("path")
    assert "page_meta" in specs
    assert page_meta.output_file == "page_meta_summary.json"
    assert page_meta.benchmark_version == "v1.0"
    assert page_meta.global_aggregation is not None
    assert page_meta.global_aggregation.method == "weighted_mean"
    assert page_meta.report is not None
    assert page_meta.report.per_field is True
    assert page_meta.normalize_config is not None
    assert page_meta.normalize_config.lowercase is True
    assert [field.field for section in page_meta.sections for field in section.fields] == [
        "entity_name",
        "page_num",
        "page_type",
        "statement_type",
        "title",
    ]
    first_field = page_meta.sections[0].fields[0]
    assert first_field.key == "meta.entity_name"
    assert first_field.compare_method == "string_similarity"
    assert first_field.metrics == ("mean",)
    assert first_field.weight == 1.0
    assert facts_value is not None
    assert facts_value.key == "facts.value"
    assert facts_value.matcher == "lcs"
    assert facts_value.alignment_on == ("value",)
    assert facts_value.aggregate_on == "facts"
    assert facts_value.metric_groups["value"].metrics == ("accuracy",)
    assert facts_value.metric_groups["value"].on == "matched_pairs"
    assert facts_value.metric_groups["alignment"].metrics == ("precision", "recall", "f1")
    assert facts_value.metric_groups["alignment"].on is None
    assert facts_value.dash_normalizer == "treat_as_zero"
    assert facts_value.special_value_normalization == {"-": "0"}
    assert facts_path is not None
    assert facts_path.compare_method == "path_all_levels_threshold"
    assert facts_path.compare_methods == ("path_all_levels_threshold", "path_last_leaf_threshold", "path_soft_overlap")
    assert facts_path.threshold == 0.85
    assert facts_path.require_same_length is True
    assert facts_path.aggregate_on == "facts"
    assert facts_path.metric_groups["strict"].metrics == ("accuracy",)
    assert facts_path.metric_groups["leaf"].metrics == ("accuracy",)
    assert facts_path.metric_groups["soft_overlap"].metrics == ("precision", "recall", "f1")
    assert facts.under_development_message == "Facts evaluation is under development."


def test_load_evaluation_specs_rejects_legacy_facts_values_key(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation_specs.yaml"
    config_path.write_text(
        json.dumps(
            {
                "benchmark_version": "v1.0",
                "global_aggregation": {"method": "weighted_mean", "normalize_weights": True},
                "report": {"per_field": True, "per_document": False, "include_std": False},
                "normalize_config": {"strip": True, "quotes_unify": True, "lowercase": True},
                "fields": {
                    "meta.entity_name": {
                        "compare_method": "string_similarity",
                        "averaging_rule": "micro",
                        "normalize": True,
                        "null_handling": "regular",
                        "metrics": ["mean"],
                        "weight": 1.0,
                    },
                    "facts.values": {
                        "matcher": "lcs",
                        "compare_method": "exact_match",
                        "metrics": {"value": {"metrics": ["accuracy"], "on": "matched_pairs"}, "alignment": ["precision", "recall", "f1"]},
                        "averaging_rule": "micro",
                        "aggregate_on": "facts",
                        "normalize": True,
                        "null_handling": "regular",
                        "dash_normalizer": "treat_as_zero",
                        "special_value_normalization": {"-": "0"},
                        "weight": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        load_evaluation_specs_from_path(config_path)
    except ValueError as exc:
        assert "facts.value" in str(exc)
    else:
        raise AssertionError("Expected legacy facts.values key to be rejected.")


def test_dataset_option_label_uses_workspace_stats() -> None:
    dataset = next(version for version in list_dataset_versions(data_root=Path("/Users/delmedigo/Dev/FineTree/data")) if version.version_id == "4ee1c00b-ef4d-4bc9-8fe3-e34ce38f9b69")
    assert dataset_option_label(dataset) == "maayan | 4ee1c00b-ef4d-4bc9-8fe3-e34ce38f9b69 | val_docs=1 | val_pages=10"


def test_list_native_inference_runs_filters_dataset_and_val_split(tmp_path: Path) -> None:
    selected = tmp_path / "outputs" / "20260410_010000_selected"
    wrong_split = tmp_path / "outputs" / "20260410_020000_wrong_split"
    wrong_dataset = tmp_path / "outputs" / "20260410_030000_wrong_dataset"
    parsed_page = {"image": "page_0001.png", "meta": {}, "facts": []}
    _write_native_run(selected, doc_id="doc1", parsed_page=parsed_page, dataset_version_id="dataset-1", model="model-new", started_at=20.0)
    _write_native_run(wrong_split, doc_id="doc1", parsed_page=parsed_page, dataset_version_id="dataset-1", split="train", model="model-train", started_at=30.0)
    _write_native_run(wrong_dataset, doc_id="doc1", parsed_page=parsed_page, dataset_version_id="dataset-2", model="model-other", started_at=40.0)
    older = tmp_path / "outputs" / "20260409_230000_selected_old"
    _write_native_run(older, doc_id="doc1", parsed_page=parsed_page, dataset_version_id="dataset-1", model="model-old", started_at=10.0)

    runs = list_native_inference_runs("dataset-1", output_root=tmp_path / "outputs")
    assert [run.run_id for run in runs] == ["20260410_010000_selected", "20260409_230000_selected_old"]
    assert [run.model_name for run in runs] == ["model-new", "model-old"]


def test_main_without_tty_requires_explicit_subcommands(monkeypatch) -> None:
    class _FakeInput:
        def isatty(self) -> bool:
            return False

    class _FakeOutput(_TTYStringIO):
        def isatty(self) -> bool:
            return False

    stderr = _FakeOutput()
    monkeypatch.setattr(sys, "stdin", _FakeInput())
    monkeypatch.setattr(sys, "stdout", _FakeOutput())
    monkeypatch.setattr(sys, "stderr", stderr)
    exit_code = main([])
    assert exit_code == 2
    assert "Interactive benchmark CLI requires a TTY." in stderr.getvalue()


def test_main_without_args_launches_interactive_when_tty(monkeypatch) -> None:
    called: dict[str, bool] = {"value": False}

    monkeypatch.setattr(sys, "stdin", _TTYStringIO())
    monkeypatch.setattr(sys, "stdout", _TTYStringIO())

    def _fake_run() -> int:
        called["value"] = True
        return 0

    monkeypatch.setattr("benchmark_new.cli.run_interactive_benchmark", _fake_run)
    assert main([]) == 0
    assert called["value"] is True


def test_run_interactive_benchmark_facts_target_prints_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    datasets_dir = data_root / "datasets"
    datasets_dir.mkdir(parents=True)
    dataset_version_id = "dataset-1"
    (datasets_dir / f"{dataset_version_id}.json").write_text(
        json.dumps(
            {
                "version_id": dataset_version_id,
                "name": "Dataset One",
                "created_at": 1,
                "updated_at": 2,
                "split_assignments": {"doc1": "val"},
                "split_stats": {"val": {"doc_count": 1, "page_count": 1}},
                "export_config": {"approved_pages_only": True},
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    _write_annotation(
        data_root,
        doc_id="doc1",
        page_payload={
            "image": "page_0001.png",
            "meta": {"entity_name": "Acme"},
            "facts": [{"fact_num": 1, "path": ["Assets"], "value": "1,000"}],
        },
    )
    parsed_page = {"image": "page_0001.png", "meta": {"entity_name": "Acme"}, "facts": [{"fact_num": 1, "path": ["Assets"], "value": "1000"}]}
    _write_native_run(output_root / "20260410_010000_model_a", doc_id="doc1", parsed_page=parsed_page, dataset_version_id=dataset_version_id, dataset_name="Dataset One")
    seen_labels: dict[str, list[str]] = {}

    def _picker(title: str, options: list[object], render_label) -> object:
        seen_labels[title] = [render_label(option) for option in options]
        if title == "Choose what to evaluate":
            return "facts"
        return options[0]

    stdout = io.StringIO()
    exit_code = run_interactive_benchmark(data_root=data_root, output_root=output_root, picker=_picker, stdout=stdout)
    assert exit_code == 0
    assert seen_labels["Choose dataset to evaluate"] == ["Dataset One | dataset-1 | val_docs=1 | val_pages=1"]
    assert seen_labels["Choose previous inference"] == ["model-a (20260410_010000_model_a)"]
    text = stdout.getvalue()
    assert "Facts" in text
    assert "Facts Evaluated: 1" in text
    assert "value" in text
    assert "path" in text
    assert "path_all_levels_threshold" in text
    assert "path_last_leaf_threshold" in text
    assert "soft_overlap" in text
    assert "alignment" in text
    assert "accuracy" in text
    assert "Facts Summary:" in text
    assert "Detailed Values Report:" in text
    assert "Simple Values Mistakes:" in text
    assert "Path Comparison Report" not in text


def test_execute_interactive_evaluation_both_writes_page_meta_and_facts_summaries(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    gt_page = {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [{"fact_num": 1, "path": ["Assets"], "value": "(*)-"}],
    }
    _write_annotation(data_root, doc_id="doc1", page_payload=gt_page)
    run_dir = tmp_path / "outputs" / "20260410_010000_model_a"
    _write_native_run(
        run_dir,
        doc_id="doc1",
        parsed_page=gt_page,
        dataset_version_id="dataset-1",
        dataset_name="Dataset One",
        model="model-a",
        started_at=10.0,
    )
    run = NativeRunInfo(
        run_id=run_dir.name,
        run_dir=run_dir,
        model_name="model-a",
        dataset_version_id="dataset-1",
        dataset_name="Dataset One",
        split="val",
        started_at=10.0,
    )
    stdout = io.StringIO()
    exit_code = execute_interactive_evaluation("both", run=run, data_root=data_root, stdout=stdout)
    assert exit_code == 0
    text = stdout.getvalue()
    assert "Page Meta" in text
    assert "entity_name" in text
    assert "Facts" in text
    assert "path" in text
    assert "path_all_levels_threshold" in text
    assert "path_last_leaf_threshold" in text
    assert "soft_overlap" in text
    assert "alignment" in text
    assert "Final Score" in text
    assert "Combined:" in text
    page_meta_artifact_path = run_dir / "evaluation" / "page_meta_summary.json"
    page_meta_payload = json.loads(page_meta_artifact_path.read_text(encoding="utf-8"))
    assert page_meta_payload["benchmark_version"] == "v1.0"
    assert page_meta_payload["global_aggregation_method"] == "weighted_mean"
    assert page_meta_payload["overall_score"] == 1.0
    assert page_meta_payload["page_count"] == 1
    assert page_meta_payload["fields"][0]["field"] == "entity_name"
    assert page_meta_payload["fields"][0]["key"] == "meta.entity_name"
    assert page_meta_payload["fields"][0]["metrics"] == ["mean"]
    assert page_meta_payload["fields"][0]["weight"] == 1.0
    assert page_meta_payload["fields"][0]["score"] == 1.0
    facts_artifact_path = run_dir / "evaluation" / "facts_summary.json"
    facts_payload = json.loads(facts_artifact_path.read_text(encoding="utf-8"))
    assert facts_payload["benchmark_version"] == "v1.0"
    assert facts_payload["overall_score"] == 1.0
    assert facts_payload["fact_count"] == 1
    assert facts_payload["channels"][0]["field"] == "value"
    assert any(channel["channel"] == "value" and channel["metric"] == "accuracy" for channel in facts_payload["channels"])
    assert any(channel["channel"] == "alignment" and channel["metric"] == "f1" for channel in facts_payload["channels"])
    assert any(channel["field"] == "path" and channel["channel"] == "path_all_levels_threshold" and channel["metric"] == "accuracy" for channel in facts_payload["channels"])
    assert any(channel["field"] == "path" and channel["channel"] == "path_last_leaf_threshold" and channel["metric"] == "accuracy" for channel in facts_payload["channels"])
    assert any(channel["field"] == "path" and channel["channel"] == "soft_overlap" and channel["metric"] == "f1" for channel in facts_payload["channels"])
    values_artifact_path = run_dir / "evaluation" / "mistakes_values.json"
    values_payload = json.loads(values_artifact_path.read_text(encoding="utf-8"))
    assert values_payload["page_count"] == 1
    page_payload = values_payload["documents"][0]["pages"][0]
    assert page_payload["gt_values"][0]["value"] == "0"
    assert page_payload["pred_values"][0]["value"] == "0"
    assert page_payload["value_matching"]["matched_pair_count"] == 1
    simple_values_artifact_path = run_dir / "evaluation" / "values_mistakes.json"
    simple_values_payload = json.loads(simple_values_artifact_path.read_text(encoding="utf-8"))
    assert simple_values_payload["mistake_count"] == 0
    comparison_artifact_path = run_dir / "evaluation" / "path_comparison_report.json"
    comparison_payload = json.loads(comparison_artifact_path.read_text(encoding="utf-8"))
    assert "counts" in comparison_payload
