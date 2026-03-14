from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from finetree_annotator import list_commands
from finetree_annotator.benchmark.config import DEFAULT_META_FIELD_WEIGHTS
from finetree_annotator.benchmark.submission import SUBMISSION_FIELD_SPECS
from finetree_annotator.benchmark import web


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
    values["checkpoint_name"] = "checkpoint-cli"
    return values


def _write_config(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    gt_path = tmp_path / "gt.json"
    gt_path.write_text('{"schema_version": 3, "images_dir": null, "metadata": {}, "pages": []}', encoding="utf-8")
    pred_path = input_dir / "pred.json"
    pred_path.write_text('{"meta": {}, "facts": []}', encoding="utf-8")
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
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_pyproject_registers_benchmark_script() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = list_commands._scripts_from_pyproject(pyproject)
    assert "finetree-benchmark" in scripts
    assert "finetree-benchmark-run" in scripts


def test_benchmark_main_boots_uvicorn(monkeypatch, tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    calls: dict[str, object] = {}

    def fake_create_app(*, config_path):
        calls["config_path"] = config_path
        return "app-instance"

    def fake_run(app, host, port, log_level):
        calls["run"] = {
            "app": app,
            "host": host,
            "port": port,
            "log_level": log_level,
        }

    monkeypatch.setattr(web, "create_app", fake_create_app)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    exit_code = web.main(["--config", str(cfg_path), "--host", "127.0.0.1", "--port", "8124"])
    assert exit_code == 0
    assert calls["config_path"] == str(cfg_path)
    assert calls["run"] == {
        "app": "app-instance",
        "host": "127.0.0.1",
        "port": 8124,
        "log_level": "info",
    }
