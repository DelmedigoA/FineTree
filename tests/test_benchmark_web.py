from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from finetree_annotator.benchmark.config import DEFAULT_META_FIELD_WEIGHTS
from finetree_annotator.benchmark.submission import SUBMISSION_FIELD_SPECS
from finetree_annotator.benchmark.web import create_app


def _sample_model_metadata() -> dict[str, object]:
    values: dict[str, object] = {}
    for spec in SUBMISSION_FIELD_SPECS:
        if spec.value_type == "bool":
            values[spec.name] = True
        elif spec.value_type == "int":
            values[spec.name] = 2
        elif spec.value_type == "float":
            values[spec.name] = 0.25
        elif spec.value_type == "json":
            values[spec.name] = {"use_reentrant": False}
        else:
            values[spec.name] = f"{spec.name}-value"
    values["checkpoint_name"] = "checkpoint-web"
    values["CUDA_VISIBLE_DEVICES"] = "0"
    values["gpu_used"] = "A100"
    values["torch_env_used"] = "torch 2.6.0"
    values["platform"] = "linux/amd64"
    return values


def _page(*, entity: str, page_num: str, page_type: str, statement_type: str | None, title: str, fact_count: int) -> dict:
    return {
        "image": "page_0001.png",
        "meta": {
            "entity_name": entity,
            "page_num": page_num,
            "page_type": page_type,
            "statement_type": statement_type,
            "title": title,
        },
        "facts": [{"value": str(index), "path": [f"fact-{index}"]} for index in range(fact_count)],
    }


def _write_config_and_inputs(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    gt_doc = {
        "schema_version": 3,
        "images_dir": None,
        "metadata": {},
        "pages": [_page(entity="Acme", page_num="1", page_type="title", statement_type=None, title="Annual", fact_count=2)],
    }
    pred_page = {
        "meta": gt_doc["pages"][0]["meta"],
        "facts": gt_doc["pages"][0]["facts"],
    }
    gt_path = tmp_path / "gt.json"
    pred_path = input_dir / "pred.json"
    gt_path.write_text(json.dumps(gt_doc, ensure_ascii=False), encoding="utf-8")
    pred_path.write_text(json.dumps(pred_page, ensure_ascii=False), encoding="utf-8")

    payload = {
        "benchmark": {
            "input_dir": str(input_dir),
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
        "model_metadata": _sample_model_metadata(),
        "mappings": [
            {
                "prediction_file": "pred.json",
                "gt_file": str(gt_path),
                "prediction_format": "page_level",
                "gt_page_index": 0,
            }
        ],
    }
    cfg_path = tmp_path / "benchmark.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg_path


def _submission_form_data() -> dict[str, str]:
    data: dict[str, str] = {}
    defaults = _sample_model_metadata()
    for spec in SUBMISSION_FIELD_SPECS:
        data[spec.name] = spec.serialize_default(defaults[spec.name])
    return data


def _write_info_submission(tmp_path: Path) -> Path:
    submission_dir = tmp_path / "submission_input"
    predictions_dir = submission_dir / "predictions"
    output_dir = tmp_path / "output"
    predictions_dir.mkdir(parents=True)
    output_dir.mkdir()

    gt_doc = {
        "schema_version": 3,
        "images_dir": None,
        "metadata": {},
        "pages": [_page(entity="Acme", page_num="1", page_type="title", statement_type=None, title="Annual", fact_count=2)],
    }
    pred_page = {"meta": gt_doc["pages"][0]["meta"], "facts": gt_doc["pages"][0]["facts"]}
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_doc, ensure_ascii=False), encoding="utf-8")
    (predictions_dir / "pred_0001.json").write_text(json.dumps(pred_page, ensure_ascii=False), encoding="utf-8")
    (submission_dir / "logging.jsonl").write_text('{"eval_loss": 0.1, "epoch": 1.0, "global_step/max_steps": "10/10"}\n', encoding="utf-8")
    info_payload = {
        "model_metadata": _sample_model_metadata(),
        "training_args": {
            "dataset": "train-ds",
            "validation_dataset": "val-ds",
        },
        "environment": {
            "platform": "linux/amd64",
            "torch_version": "2.6.0",
        },
        "run": {
            "run_id": "run-123",
            "training_started_at_israel": "2026-03-14T10:00:00+02:00",
        },
        "selected_checkpoint": {
            "checkpoint_name": "checkpoint-10",
            "epoch": 1.0,
            "eval_loss": 0.1,
            "pushed_adapter_to_hub": True,
            "hub_model_id": "asafd60/model-adapter-best",
        },
        "artifacts": {
            "logging_jsonl": "logging.jsonl",
            "prediction_dir": "predictions",
            "benchmark_report": "report.json",
        },
    }
    (submission_dir / "info.json").write_text(
        json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "benchmark-info.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "input_dir": str(submission_dir),
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
    return cfg_path


def test_benchmark_web_submission_and_leaderboard_flow(tmp_path: Path) -> None:
    cfg_path = _write_config_and_inputs(tmp_path)
    client = TestClient(create_app(config_path=cfg_path))

    submission_page = client.get("/submission")
    assert submission_page.status_code == 200
    assert "Benchmark Submission" in submission_page.text

    leaderboard_page = client.get("/leaderboard")
    assert leaderboard_page.status_code == 200
    assert "Benchmark Leaderboard" in leaderboard_page.text

    config_payload = client.get("/api/config")
    assert config_payload.status_code == 200
    assert config_payload.json()["mapping_checks"][0]["status"] == "ok"

    logging_text = (
        '{"loss": 0.5, "token_acc": 0.9, "learning_rate": 0.001, "epoch": 1.0, "global_step/max_steps": "1/2", "memory(GiB)": 10.0, "train_speed(s/it)": 2.0}\n'
        '{"eval_loss": 0.4, "eval_runtime": 12.0, "eval_token_acc": 0.92, "epoch": 1.0, "global_step/max_steps": "1/2", "elapsed_time": "1m", "remaining_time": "1m", "memory(GiB)": 10.5}\n'
    )
    response = client.post(
        "/api/submissions",
        data=_submission_form_data(),
        files={"logging_jsonl": ("logging.jsonl", logging_text.encode("utf-8"), "application/json")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    submission_dir = Path(payload["submission_dir"])
    report_path = submission_dir / "report.json"
    logging_path = submission_dir / "logging.jsonl"
    assert report_path.is_file()
    assert logging_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["aggregate_metrics"]["overall_score"] == 1.0
    assert report["aggregate_metrics"]["facts_score"] == 1.0
    assert report["leaderboard_row"]["checkpoint_name"] == "checkpoint-web"
    assert report["report_timestamp_israel"].endswith(("+02:00", "+03:00"))

    leaderboard = client.get("/api/leaderboard")
    assert leaderboard.status_code == 200
    leaderboard_payload = leaderboard.json()
    assert leaderboard_payload["columns"] == [
        "model",
        "report_timestamp_israel",
        "dataset",
        "overall_score",
        "meta_score",
        "facts_score",
        "meta_hard_score",
        "entity_score",
        "title_score",
        "page_num_score",
        "page_type_score",
        "statement_type_score",
        "facts_count_score",
    ]
    rows = leaderboard_payload["rows"]
    assert len(rows) == 1
    assert rows[0]["checkpoint_name"] == "checkpoint-web"
    assert rows[0]["_details"]["submission_metadata"]["gpu_used"] == "A100"


def test_benchmark_submission_lock_rejects_concurrent_runs(tmp_path: Path) -> None:
    cfg_path = _write_config_and_inputs(tmp_path)
    app = create_app(config_path=cfg_path)
    client = TestClient(app)
    assert app.state.submission_lock.acquire(blocking=False) is True
    try:
        response = client.post(
            "/api/submissions",
            data=_submission_form_data(),
            files={"logging_jsonl": ("logging.jsonl", b'{"loss": 1}\n', "application/json")},
        )
    finally:
        app.state.submission_lock.release()
    assert response.status_code == 409


def test_benchmark_web_uses_info_json_submission_when_present(tmp_path: Path) -> None:
    cfg_path = _write_info_submission(tmp_path)
    client = TestClient(create_app(config_path=cfg_path))

    config_payload = client.get("/api/config")
    assert config_payload.status_code == 200
    info_submission = config_payload.json()["info_submission"]
    assert info_submission["status"] == "ready"
    assert info_submission["metadata"]["checkpoint_name"] == "checkpoint-web"

    response = client.post("/api/submissions")
    assert response.status_code == 200, response.text
    payload = response.json()
    submission_dir = Path(payload["submission_dir"])
    report = json.loads((submission_dir / "report.json").read_text(encoding="utf-8"))
    mirrored_report = json.loads((tmp_path / "submission_input" / "report.json").read_text(encoding="utf-8"))
    assert report["submission_source"] == "info_json"
    assert report["submission_context"]["run"]["run_id"] == "run-123"
    assert report["leaderboard_row"]["selected_checkpoint_checkpoint_name"] == "checkpoint-10"
    assert mirrored_report["submission_id"] == report["submission_id"]

    leaderboard = client.get("/api/leaderboard")
    assert leaderboard.status_code == 200
    payload = leaderboard.json()
    assert payload["rows"][0]["_details"]["submission_context"]["selected_checkpoint"]["hub_model_id"] == "asafd60/model-adapter-best"


def test_benchmark_web_can_delete_submission(tmp_path: Path) -> None:
    cfg_path = _write_config_and_inputs(tmp_path)
    client = TestClient(create_app(config_path=cfg_path))

    response = client.post(
        "/api/submissions",
        data=_submission_form_data(),
        files={"logging_jsonl": ("logging.jsonl", b'{"loss": 0.1}\n', "application/json")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    submission_id = payload["submission_id"]
    submission_dir = Path(payload["submission_dir"])
    assert submission_dir.is_dir()

    delete_response = client.delete(f"/api/submissions/{submission_id}")
    assert delete_response.status_code == 200, delete_response.text
    assert not submission_dir.exists()

    leaderboard = client.get("/api/leaderboard")
    assert leaderboard.status_code == 200
    assert leaderboard.json()["rows"] == []
