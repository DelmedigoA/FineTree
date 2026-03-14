from __future__ import annotations

import json
from pathlib import Path

import yaml

from finetree_annotator.benchmark.config import DEFAULT_META_FIELD_WEIGHTS
from finetree_annotator.benchmark.notebook_support import (
    build_benchmark_model_metadata,
    build_submission_info_payload,
    materialize_prediction_json_files,
    select_best_checkpoint,
)
from finetree_annotator.benchmark.runner import run_info_submission
from finetree_annotator.benchmark.info import load_submission_info, load_submission_info_bundle
from finetree_annotator.benchmark.submission import SUBMISSION_FIELD_SPECS


def _sample_model_metadata() -> dict[str, object]:
    values: dict[str, object] = {}
    for spec in SUBMISSION_FIELD_SPECS:
        if spec.value_type == "bool":
            values[spec.name] = True
        elif spec.value_type == "int":
            values[spec.name] = 1
        elif spec.value_type == "float":
            values[spec.name] = 0.25
        elif spec.value_type == "json":
            values[spec.name] = {"use_reentrant": False}
        else:
            values[spec.name] = f"{spec.name}-value"
    values["checkpoint_name"] = "checkpoint-10"
    values["dataset"] = "train-ds"
    values["validation_dataset"] = "val-ds"
    values["gpu_used"] = "A100"
    values["torch_env_used"] = "torch 2.6.0"
    values["platform"] = "linux/amd64"
    return values


def _page() -> dict[str, object]:
    return {
        "image": "page_0001.png",
        "meta": {
            "entity_name": "Acme",
            "page_num": "1",
            "page_type": "title",
            "statement_type": None,
            "title": "Annual",
        },
        "facts": [{"value": "1", "path": ["fact-1"]}],
    }


def test_select_best_checkpoint_uses_lowest_eval_loss(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "checkpoint-10").mkdir(parents=True)
    (output_dir / "checkpoint-20").mkdir()
    rows = [
        {"eval_loss": 0.3, "epoch": 1.0, "global_step/max_steps": "10/20"},
        {"eval_loss": 0.2, "epoch": 2.0, "global_step/max_steps": "20/20"},
    ]
    best = select_best_checkpoint(output_dir, rows)
    assert best["checkpoint_name"] == "checkpoint-20"
    assert best["eval_loss"] == 0.2


def test_materialize_prediction_json_files_writes_sequential_json(tmp_path: Path) -> None:
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        "\n".join(
            [
                json.dumps({"response": json.dumps({"meta": {"entity_name": "A"}, "facts": []})}),
                json.dumps({"prediction": {"meta": {"entity_name": "B"}, "facts": []}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written = materialize_prediction_json_files(result_path, tmp_path / "predictions")
    assert [path.name for path in written] == ["pred_0001.json", "pred_0002.json"]
    first = json.loads(written[0].read_text(encoding="utf-8"))
    assert first["meta"]["entity_name"] == "A"


def test_materialize_prediction_json_files_rejects_label_fallback(tmp_path: Path) -> None:
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps(
            {
                "response": '{"meta":{"entity_name":"Broken"}',
                "labels": json.dumps({"meta": {"entity_name": "GT"}, "facts": []}),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        materialize_prediction_json_files(result_path, tmp_path / "predictions")
    except ValueError as exc:
        assert "Refusing to fall back to labels" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected materialize_prediction_json_files to reject label fallback.")


def test_run_info_submission_persists_report_and_mirror(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    predictions_dir = submission_dir / "predictions"
    output_dir = tmp_path / "output"
    predictions_dir.mkdir(parents=True)
    output_dir.mkdir()

    gt_doc = {
        "schema_version": 3,
        "images_dir": None,
        "metadata": {},
        "pages": [_page()],
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_doc, ensure_ascii=False), encoding="utf-8")
    (predictions_dir / "pred_0001.json").write_text(
        json.dumps({"meta": gt_doc["pages"][0]["meta"], "facts": gt_doc["pages"][0]["facts"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (submission_dir / "logging.jsonl").write_text(
        '{"eval_loss": 0.1, "eval_token_acc": 0.9, "epoch": 1.0, "global_step/max_steps": "10/10"}\n',
        encoding="utf-8",
    )
    metadata = _sample_model_metadata()
    info_payload = build_submission_info_payload(
        model_metadata=metadata,
        training_args={"dataset": "train-ds", "validation_dataset": "val-ds"},
        environment={"platform": "linux/amd64", "torch_version": "2.6.0", "gpu_used": "A100"},
        run={"run_id": "run-1"},
        selected_checkpoint={"checkpoint_name": "checkpoint-10", "epoch": 1.0, "eval_loss": 0.1},
        artifacts={"logging_jsonl": "logging.jsonl", "prediction_dir": "predictions"},
    )
    (submission_dir / "info.json").write_text(
        json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cfg_path = tmp_path / "benchmark.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "input_dir": str(tmp_path / "unused-input"),
                    "output_dir": str(output_dir),
                    "timezone": "Asia/Jerusalem",
                },
                "methods": {
                    "meta": "weighted_soft_hard",
                    "facts": "count_only",
                    "overall": "weighted_average",
                },
                "weighting": {
                    "meta_fields": DEFAULT_META_FIELD_WEIGHTS,
                    "aggregate": {"meta_score": 1, "facts_score": 1},
                },
                "evaluation": {
                    "normalize_inputs": True,
                    "prediction_format_default": "auto",
                    "require_explicit_mappings": True,
                },
                "mappings": [
                    {
                        "prediction_file": "predictions/pred_0001.json",
                        "gt_file": str(gt_path),
                        "prediction_format": "page_level",
                        "gt_page_index": 0,
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    report, persisted_dir = run_info_submission(config_path=cfg_path, submission_dir=submission_dir)
    assert report.aggregate_metrics["overall_score"] == 1.0
    assert (persisted_dir / "report.json").is_file()
    assert (submission_dir / "report.json").is_file()
    mirrored = json.loads((submission_dir / "report.json").read_text(encoding="utf-8"))
    assert mirrored["submission_source"] == "info_json"


def test_build_benchmark_model_metadata_populates_required_fields() -> None:
    metadata = build_benchmark_model_metadata(
        training_args={
            "model": "Qwen/Qwen3.5-27B",
            "dataset": "train-ds",
            "validation_dataset": "val-ds",
            "tuner_type": "lora",
            "use_hf": True,
        },
        environment={
            "platform": "linux/amd64",
            "torch_version": "2.6.0",
            "cuda_runtime_version": "12.4",
            "python_version": "3.12.2",
            "cuda_visible_devices": "0",
            "max_pixels_env": "1000000",
            "gpu_used": "A100",
        },
        checkpoint_name="checkpoint-best",
    )
    assert metadata["checkpoint_name"] == "checkpoint-best"
    assert metadata["validation_dataset"] == "val-ds"
    assert metadata["gpu_used"] == "A100"
    assert metadata["platform"] == "linux/amd64"


def test_load_submission_info_accepts_integral_float_metadata_fields(tmp_path: Path) -> None:
    info_path = tmp_path / "info.json"
    payload = {
        "model_metadata": {
            **_sample_model_metadata(),
            "eval_steps": 5.0,
            "save_steps": 5.0,
            "logging_steps": 5.0,
        },
        "training_args": {},
        "environment": {},
        "run": {},
        "selected_checkpoint": {},
        "artifacts": {},
    }
    info_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info = load_submission_info(info_path)
    assert info.model_metadata["eval_steps"] == 5
    assert info.model_metadata["save_steps"] == 5
    assert info.model_metadata["logging_steps"] == 5


def test_load_submission_info_bundle_rejects_predictions_not_matching_raw_response(tmp_path: Path) -> None:
    submission_dir = tmp_path / "submission"
    predictions_dir = submission_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    (submission_dir / "logging.jsonl").write_text('{"loss": 0.1}\n', encoding="utf-8")
    (submission_dir / "raw_predictions.jsonl").write_text(
        json.dumps({"response": json.dumps({"meta": {"entity_name": "model"}, "facts": []})}) + "\n",
        encoding="utf-8",
    )
    (predictions_dir / "pred_0001.json").write_text(
        json.dumps({"meta": {"entity_name": "gt"}, "facts": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    info_payload = build_submission_info_payload(
        model_metadata=_sample_model_metadata(),
        training_args={"dataset": "train-ds", "validation_dataset": "val-ds"},
        environment={"platform": "linux/amd64"},
        run={"run_id": "run-1"},
        selected_checkpoint={"checkpoint_name": "checkpoint-10"},
        artifacts={
            "logging_jsonl": "logging.jsonl",
            "prediction_dir": "predictions",
            "raw_predictions_jsonl": "raw_predictions.jsonl",
        },
    )
    (submission_dir / "info.json").write_text(
        json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        load_submission_info_bundle(submission_dir)
    except ValueError as exc:
        assert "does not match model response" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected bundle validation to reject predictions that do not match raw response.")
