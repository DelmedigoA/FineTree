from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .logging_summary import parse_logging_jsonl_bytes
from .notebook_support import extract_prediction_payload
from .submission import validate_submission_defaults


class SubmissionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_metadata: dict[str, Any]
    training_args: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    selected_checkpoint: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_metadata", mode="before")
    @classmethod
    def _validate_model_metadata(cls, value: Any) -> dict[str, Any]:
        return validate_submission_defaults(value)

    @field_validator("training_args", "environment", "run", "selected_checkpoint", "artifacts", mode="before")
    @classmethod
    def _validate_mapping_sections(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Submission info sections must be JSON objects.")
        return {str(key): section_value for key, section_value in value.items()}


@dataclass(frozen=True)
class SubmissionInfoBundle:
    submission_dir: Path
    info_path: Path
    logging_path: Path
    logging_text: str
    logging_rows: list[dict[str, Any]]
    info: SubmissionInfo

    @property
    def context(self) -> dict[str, Any]:
        payload = self.info.model_dump(mode="json")
        payload.pop("model_metadata", None)
        return payload


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def load_submission_info(info_path: Path | str) -> SubmissionInfo:
    path = Path(info_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Submission info not found: {path}")
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Submission info must be a JSON object: {path}")
    return SubmissionInfo.model_validate(payload)


def _resolve_submission_path(root: Path, raw_value: Any, *, default_name: str) -> Path:
    text = str(raw_value).strip() if raw_value not in (None, "") else default_name
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return candidate


def _validate_raw_prediction_bundle(root: Path, info: SubmissionInfo) -> None:
    raw_predictions_path = _resolve_submission_path(
        root,
        info.artifacts.get("raw_predictions_jsonl"),
        default_name="raw_predictions.jsonl",
    )
    if not raw_predictions_path.is_file():
        return
    prediction_dir = _resolve_submission_path(
        root,
        info.artifacts.get("prediction_dir"),
        default_name="predictions",
    )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found for submission: {prediction_dir}")

    prediction_files = sorted(prediction_dir.glob("pred_*.json"))
    raw_rows = [json.loads(line) for line in raw_predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prediction_files) != len(raw_rows):
        raise ValueError(
            f"Prediction file count {len(prediction_files)} does not match raw prediction row count {len(raw_rows)}."
        )

    for index, (prediction_path, row) in enumerate(zip(prediction_files, raw_rows, strict=True), start=1):
        try:
            expected_prediction = extract_prediction_payload(row)
        except Exception as exc:
            raise ValueError(
                f"Could not validate raw prediction row {index} from {raw_predictions_path.name}: {exc}"
            ) from exc
        actual_prediction = _read_json_file(prediction_path)
        if actual_prediction != expected_prediction:
            raise ValueError(
                f"Prediction file {prediction_path.name} does not match model response in {raw_predictions_path.name} "
                f"at row {index}."
            )


def load_submission_info_bundle(submission_dir: Path | str) -> SubmissionInfoBundle:
    root = Path(submission_dir).expanduser().resolve()
    info_path = root / "info.json"
    info = load_submission_info(info_path)
    _validate_raw_prediction_bundle(root, info)
    logging_path = _resolve_submission_path(root, info.artifacts.get("logging_jsonl"), default_name="logging.jsonl")
    if not logging_path.is_file():
        raise FileNotFoundError(f"logging.jsonl not found for submission: {logging_path}")
    logging_text, logging_rows = parse_logging_jsonl_bytes(logging_path.read_bytes())
    return SubmissionInfoBundle(
        submission_dir=root,
        info_path=info_path,
        logging_path=logging_path,
        logging_text=logging_text,
        logging_rows=logging_rows,
        info=info,
    )


def discover_submission_info(submission_dir: Path | str) -> dict[str, Any] | None:
    root = Path(submission_dir).expanduser().resolve()
    info_path = root / "info.json"
    if not info_path.is_file():
        return None
    try:
        bundle = load_submission_info_bundle(root)
    except Exception as exc:
        return {
            "mode": "info_json",
            "status": "error",
            "submission_dir": str(root),
            "info_path": str(info_path),
            "errors": [str(exc)],
        }
    return {
        "mode": "info_json",
        "status": "ready",
        "submission_dir": str(bundle.submission_dir),
        "info_path": str(bundle.info_path),
        "logging_path": str(bundle.logging_path),
        "metadata": bundle.info.model_metadata,
        "run": bundle.info.run,
        "selected_checkpoint": bundle.info.selected_checkpoint,
        "artifacts": bundle.info.artifacts,
        "errors": [],
    }
