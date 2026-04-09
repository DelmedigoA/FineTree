from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark_new.cli import main
from benchmark_new.eval import evaluate_single_page
from benchmark_new.models import DocumentResult, RunResult
from benchmark_new.io.datasets import list_dataset_versions, load_dataset_selection
from benchmark_new.io.run_adapters import load_predictions_from_run_dir
from benchmark_new.reports import write_full_metrics_csv, write_mistakes_json


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
    with (run_dir / "evaluation" / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "meta__entity_name__score" in rows[0]
    assert "facts__value__f1" in rows[0]
    assert "facts__path__score" in rows[0]
    mistakes = json.loads((run_dir / "evaluation" / "mistakes.json").read_text(encoding="utf-8"))
    assert mistakes["documents"] == []
