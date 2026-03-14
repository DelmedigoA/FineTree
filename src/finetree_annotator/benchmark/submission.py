from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SubmissionFieldSpec:
    name: str
    label: str
    kind: str
    value_type: str
    step: str | None = None

    def serialize_default(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    def to_api(self, default_value: Any) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "value_type": self.value_type,
            "required": True,
            "default": self.serialize_default(default_value),
        }
        if self.step is not None:
            payload["step"] = self.step
        return payload


SUBMISSION_FIELD_SPECS: tuple[SubmissionFieldSpec, ...] = (
    SubmissionFieldSpec("checkpoint_name", "Checkpoint Name", "text", "string"),
    SubmissionFieldSpec("model", "Model", "text", "string"),
    SubmissionFieldSpec("tuner_type", "Tuner Type", "text", "string"),
    SubmissionFieldSpec("dataset", "Dataset", "text", "string"),
    SubmissionFieldSpec("validation_dataset", "Validation Dataset", "text", "string"),
    SubmissionFieldSpec("use_hf", "Use HF", "boolean", "bool"),
    SubmissionFieldSpec("freeze_vit", "Freeze ViT", "boolean", "bool"),
    SubmissionFieldSpec("vit_lr", "ViT LR", "number", "float", step="any"),
    SubmissionFieldSpec("freeze_aligner", "Freeze Aligner", "boolean", "bool"),
    SubmissionFieldSpec("enable_thinking", "Enable Thinking", "boolean", "bool"),
    SubmissionFieldSpec("add_non_thinking_prefix", "Add Non Thinking Prefix", "boolean", "bool"),
    SubmissionFieldSpec("torch_dtype", "Torch DType", "text", "string"),
    SubmissionFieldSpec("num_train_epochs", "Num Train Epochs", "number", "float", step="any"),
    SubmissionFieldSpec("per_device_train_batch_size", "Train Batch Size", "number", "int", step="1"),
    SubmissionFieldSpec("per_device_eval_batch_size", "Eval Batch Size", "number", "int", step="1"),
    SubmissionFieldSpec("learning_rate", "Learning Rate", "number", "float", step="any"),
    SubmissionFieldSpec("lora_rank", "LoRA Rank", "number", "int", step="1"),
    SubmissionFieldSpec("lora_alpha", "LoRA Alpha", "number", "int", step="1"),
    SubmissionFieldSpec("target_modules", "Target Modules", "text", "string"),
    SubmissionFieldSpec("gradient_accumulation_steps", "Gradient Accumulation Steps", "number", "int", step="1"),
    SubmissionFieldSpec("output_dir", "Output Dir", "text", "string"),
    SubmissionFieldSpec("eval_steps", "Eval Steps", "number", "int", step="1"),
    SubmissionFieldSpec("save_steps", "Save Steps", "number", "int", step="1"),
    SubmissionFieldSpec("save_total_limit", "Save Total Limit", "number", "int", step="1"),
    SubmissionFieldSpec("logging_steps", "Logging Steps", "number", "int", step="1"),
    SubmissionFieldSpec("max_length", "Max Length", "number", "int", step="1"),
    SubmissionFieldSpec("warmup_ratio", "Warmup Ratio", "number", "float", step="any"),
    SubmissionFieldSpec("dataset_num_proc", "Dataset Num Proc", "number", "int", step="1"),
    SubmissionFieldSpec("dataloader_num_workers", "Dataloader Num Workers", "number", "int", step="1"),
    SubmissionFieldSpec("eval_on_start", "Eval On Start", "boolean", "bool"),
    SubmissionFieldSpec("max_pixels", "Max Pixels", "number", "int", step="1"),
    SubmissionFieldSpec("temperature", "Temperature", "number", "float", step="any"),
    SubmissionFieldSpec("gradient_checkpointing", "Gradient Checkpointing", "boolean", "bool"),
    SubmissionFieldSpec("gradient_checkpointing_kwargs", "Gradient Checkpointing Kwargs", "textarea", "json"),
    SubmissionFieldSpec("truncation_strategy", "Truncation Strategy", "text", "string"),
    SubmissionFieldSpec("CUDA_VISIBLE_DEVICES", "CUDA Visible Devices", "text", "string"),
    SubmissionFieldSpec("MAX_PIXELS", "MAX PIXELS", "number", "int", step="1"),
    SubmissionFieldSpec("gpu_used", "GPU Used", "text", "string"),
    SubmissionFieldSpec("torch_env_used", "Torch Env Used", "textarea", "string"),
    SubmissionFieldSpec("platform", "Platform", "text", "string"),
)

SUBMISSION_FIELD_NAMES: tuple[str, ...] = tuple(spec.name for spec in SUBMISSION_FIELD_SPECS)
_SPEC_BY_NAME: dict[str, SubmissionFieldSpec] = {spec.name: spec for spec in SUBMISSION_FIELD_SPECS}


def _normalize_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _normalize_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer value.")
    text = str(value).strip().replace("_", "")
    if not text:
        return None
    try:
        parsed_float = float(text)
    except ValueError:
        parsed_float = None
    if parsed_float is not None:
        if parsed_float.is_integer():
            return int(parsed_float)
        raise ValueError(f"{field_name} must be an integer value.")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer value.") from exc


def _normalize_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a numeric value.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("_", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a numeric value.") from exc


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_submission_value(spec: SubmissionFieldSpec, value: Any) -> Any:
    if spec.value_type == "bool":
        return _normalize_bool(value, field_name=spec.name)
    if spec.value_type == "int":
        return _normalize_int(value, field_name=spec.name)
    if spec.value_type == "float":
        return _normalize_float(value, field_name=spec.name)
    if spec.value_type == "json":
        return _normalize_jsonish(value)
    return _normalize_string(value)


def normalize_submission_defaults(payload: Any, *, require_all: bool) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ValueError("model_metadata must be a YAML mapping.")
    extras = [name for name in payload.keys() if name not in _SPEC_BY_NAME]
    if extras:
        raise ValueError("model_metadata contains unsupported keys: " + ", ".join(sorted(str(name) for name in extras)))
    missing = [name for name in SUBMISSION_FIELD_NAMES if name not in payload]
    if require_all and missing:
        raise ValueError("model_metadata is missing required keys: " + ", ".join(missing))
    normalized: dict[str, Any] = {}
    for spec in SUBMISSION_FIELD_SPECS:
        normalized[spec.name] = normalize_submission_value(spec, payload.get(spec.name))
    return normalized


def validate_submission_defaults(payload: Any) -> dict[str, Any]:
    return normalize_submission_defaults(payload, require_all=True)


def normalize_submission_defaults_partial(payload: Any) -> dict[str, Any]:
    return normalize_submission_defaults(payload, require_all=False)


def parse_submission_form_data(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for spec in SUBMISSION_FIELD_SPECS:
        raw_value = payload.get(spec.name)
        value = normalize_submission_value(spec, raw_value)
        if value is None:
            missing.append(spec.name)
            continue
        normalized[spec.name] = value
    if missing:
        raise ValueError("Missing required submission fields: " + ", ".join(missing))
    return normalized


def submission_fields_for_api(defaults: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [spec.to_api(defaults.get(spec.name)) for spec in SUBMISSION_FIELD_SPECS]
