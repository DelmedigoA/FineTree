from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, BenchmarkMappingConfig
from ..schema_io import load_any_schema
from ..schemas import CURRENT_SCHEMA_VERSION


@dataclass(frozen=True)
class PreparedMapping:
    mapping: BenchmarkMappingConfig
    prediction_path: Path
    gt_path: Path
    prediction_format: str
    prediction_document: dict[str, Any]
    ground_truth_document: dict[str, Any]


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def detect_document_format(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")
    if isinstance(payload.get("pages"), list):
        return "canonical_document"
    if isinstance(payload.get("facts"), list):
        return "page_level"
    raise ValueError("JSON payload must be either a canonical document or a page-level object.")


def _wrap_page_level_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "images_dir": None,
        "metadata": {},
        "pages": [
            {
                "image": str(payload.get("image") or "page_0001.png"),
                "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
                "facts": payload.get("facts") if isinstance(payload.get("facts"), list) else [],
            }
        ],
    }


def _coerce_document_payload(payload: Any, *, normalize_inputs: bool) -> dict[str, Any]:
    fmt = detect_document_format(payload)
    if fmt == "page_level":
        document_payload = _wrap_page_level_payload(payload)
    else:
        document_payload = payload
    if normalize_inputs:
        return load_any_schema(document_payload)
    if not isinstance(document_payload, dict) or not isinstance(document_payload.get("pages"), list):
        raise ValueError("Document payload must contain a pages list when normalize_inputs is false.")
    return document_payload


def _select_gt_document(document: dict[str, Any], *, gt_page_index: int | None) -> dict[str, Any]:
    if gt_page_index is None:
        return document
    pages = document.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Ground-truth document does not contain pages.")
    if gt_page_index < 0 or gt_page_index >= len(pages):
        raise ValueError(f"gt_page_index {gt_page_index} is out of range for ground-truth document.")
    return {
        "schema_version": document.get("schema_version", CURRENT_SCHEMA_VERSION),
        "images_dir": document.get("images_dir"),
        "metadata": document.get("metadata", {}),
        "pages": [pages[gt_page_index]],
    }


def _resolve_prediction_format(
    payload: Any,
    *,
    configured_format: str | None,
    default_format: str,
) -> str:
    detected = detect_document_format(payload)
    effective = configured_format or default_format
    if effective == "auto":
        return detected
    if effective != detected:
        raise ValueError(f"Configured prediction format {effective} does not match detected format {detected}.")
    return effective


def prepare_mapping(cfg: BenchmarkConfig, mapping: BenchmarkMappingConfig) -> PreparedMapping:
    prediction_path = cfg.benchmark.input_dir / mapping.prediction_file
    gt_path = mapping.gt_file
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    prediction_payload = _read_json_file(prediction_path)
    gt_payload = _read_json_file(gt_path)
    prediction_format = _resolve_prediction_format(
        prediction_payload,
        configured_format=mapping.prediction_format,
        default_format=cfg.evaluation.prediction_format_default,
    )
    if prediction_format == "page_level" and mapping.gt_page_index is None:
        raise ValueError(f"Mapping {mapping.prediction_file} requires gt_page_index for page-level prediction input.")
    if prediction_format == "canonical_document" and mapping.gt_page_index is not None:
        raise ValueError(f"Mapping {mapping.prediction_file} must not set gt_page_index for document predictions.")

    prediction_document = _coerce_document_payload(prediction_payload, normalize_inputs=cfg.evaluation.normalize_inputs)
    gt_document = _coerce_document_payload(gt_payload, normalize_inputs=cfg.evaluation.normalize_inputs)
    gt_document = _select_gt_document(gt_document, gt_page_index=mapping.gt_page_index)
    return PreparedMapping(
        mapping=mapping,
        prediction_path=prediction_path,
        gt_path=gt_path,
        prediction_format=prediction_format,
        prediction_document=prediction_document,
        ground_truth_document=gt_document,
    )


def validate_mapping_files(cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for mapping in cfg.mappings:
        prediction_path = cfg.benchmark.input_dir / mapping.prediction_file
        gt_path = mapping.gt_file
        errors: list[str] = []
        detected_format: str | None = None
        if not prediction_path.is_file():
            errors.append(f"Prediction file not found: {prediction_path}")
        if not gt_path.is_file():
            errors.append(f"Ground-truth file not found: {gt_path}")
        if prediction_path.is_file():
            try:
                payload = _read_json_file(prediction_path)
                detected_format = detect_document_format(payload)
                configured_format = mapping.prediction_format or cfg.evaluation.prediction_format_default
                if configured_format != "auto" and configured_format != detected_format:
                    errors.append(
                        f"Configured prediction format {configured_format} does not match detected format {detected_format}."
                    )
                if detected_format == "page_level" and mapping.gt_page_index is None:
                    errors.append("Page-level prediction mappings require gt_page_index.")
                if detected_format == "canonical_document" and mapping.gt_page_index is not None:
                    errors.append("Document predictions must not set gt_page_index.")
            except Exception as exc:
                errors.append(str(exc))
        results.append(
            {
                "prediction_file": str(mapping.prediction_file),
                "prediction_path": str(prediction_path),
                "gt_file": str(mapping.gt_file),
                "gt_path": str(gt_path),
                "configured_prediction_format": mapping.prediction_format or cfg.evaluation.prediction_format_default,
                "detected_prediction_format": detected_format,
                "gt_page_index": mapping.gt_page_index,
                "prediction_exists": prediction_path.is_file(),
                "gt_exists": gt_path.is_file(),
                "status": "ok" if not errors else "error",
                "errors": errors,
            }
        )
    return results
