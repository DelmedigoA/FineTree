from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finetree_annotator.benchmark.config import DEFAULT_META_FIELD_WEIGHTS, load_benchmark_config
from finetree_annotator.benchmark.submission import SUBMISSION_FIELD_SPECS


def _sample_model_metadata() -> dict[str, object]:
    values: dict[str, object] = {}
    for spec in SUBMISSION_FIELD_SPECS:
        if spec.value_type == "bool":
            values[spec.name] = True
        elif spec.value_type == "int":
            values[spec.name] = 1
        elif spec.value_type == "float":
            values[spec.name] = 0.5
        elif spec.value_type == "json":
            values[spec.name] = {"use_reentrant": False}
        else:
            values[spec.name] = f"{spec.name}-value"
    values["checkpoint_name"] = "checkpoint-a"
    return values


def _write_config(tmp_path: Path, *, mappings: list[dict[str, object]] | None = None) -> Path:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
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
            "aggregate": {
                "meta_score": 3,
                "facts_score": 1,
            },
        },
        "evaluation": {
            "normalize_inputs": True,
            "prediction_format_default": "auto",
            "require_explicit_mappings": True,
        },
        "model_metadata": _sample_model_metadata(),
        "mappings": mappings
        or [
            {
                "prediction_file": "pred.json",
                "gt_file": str(tmp_path / "gt.json"),
                "prediction_format": "page_level",
                "gt_page_index": 0,
            }
        ],
    }
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_load_benchmark_config_normalizes_aggregate_weights(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    cfg = load_benchmark_config(path)
    assert cfg.weighting.aggregate == {"meta_score": 0.75, "facts_score": 0.25}
    assert cfg.model_metadata["checkpoint_name"] == "checkpoint-a"


def test_page_level_mapping_requires_gt_page_index(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        mappings=[
            {
                "prediction_file": "pred.json",
                "gt_file": str(tmp_path / "gt.json"),
                "prediction_format": "page_level",
            }
        ],
    )
    with pytest.raises(ValueError, match="gt_page_index"):
        load_benchmark_config(path)


def test_model_metadata_defaults_can_be_partial(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    del payload["model_metadata"]["platform"]
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    cfg = load_benchmark_config(path)
    assert cfg.model_metadata["platform"] is None


def test_model_metadata_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["model_metadata"]["unknown_key"] = "x"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported keys"):
        load_benchmark_config(path)


def test_load_benchmark_config_accepts_fact_quality_v1(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["methods"]["facts"] = "fact_quality_v1"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    cfg = load_benchmark_config(path)
    assert cfg.methods.facts == "fact_quality_v1"
