from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .submission import normalize_submission_defaults_partial

DEFAULT_META_FIELD_WEIGHTS: dict[str, dict[str, float]] = {
    "meta.entity_name": {"soft": 0.75, "hard": 0.25},
    "meta.page_num": {"hard": 1.0},
    "meta.page_type": {"hard": 1.0},
    "meta.statement_type": {"hard": 1.0},
    "meta.title": {"soft": 0.75, "hard": 0.25},
}
DEFAULT_AGGREGATE_WEIGHTS: dict[str, float] = {"meta_score": 0.5, "facts_score": 0.5}
_ALLOWED_META_FIELDS = tuple(DEFAULT_META_FIELD_WEIGHTS.keys())


def _normalize_meta_rule(rule: Any) -> dict[str, float]:
    if isinstance(rule, str):
        lowered = rule.strip().lower()
        if lowered == "hard":
            return {"hard": 1.0, "soft": 0.0}
        if lowered == "soft":
            return {"hard": 0.0, "soft": 1.0}
        raise ValueError(f"Unsupported meta scoring rule: {rule}")
    if not isinstance(rule, dict):
        raise ValueError(f"Meta scoring rule must be a string or mapping, got {type(rule)!r}.")
    hard = float(rule.get("hard", 0.0))
    soft = float(rule.get("soft", 0.0))
    if hard < 0.0 or soft < 0.0:
        raise ValueError("Meta scoring weights must be non-negative.")
    total = hard + soft
    if total <= 0.0:
        raise ValueError("Meta scoring weights cannot both be zero.")
    return {"hard": hard / total, "soft": soft / total}


def _normalize_aggregate_weights(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("weighting.aggregate must be a mapping.")
    required = {"meta_score", "facts_score"}
    missing = sorted(required - set(raw.keys()))
    extras = sorted(set(raw.keys()) - required)
    if missing:
        raise ValueError("weighting.aggregate is missing required keys: " + ", ".join(missing))
    if extras:
        raise ValueError("weighting.aggregate contains unsupported keys: " + ", ".join(extras))
    weights = {key: float(raw.get(key, 0.0)) for key in required}
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("weighting.aggregate values must be non-negative.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("weighting.aggregate values cannot all be zero.")
    return {key: value / total for key, value in weights.items()}


class BenchmarkPathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    output_dir: Path
    timezone: str = "Asia/Jerusalem"

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("benchmark.timezone is required.")
        ZoneInfo(text)
        if text != "Asia/Jerusalem":
            raise ValueError("benchmark.timezone must be Asia/Jerusalem for benchmark reports.")
        return text


class BenchmarkMethodsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: Literal["weighted_soft_hard"]
    facts: Literal["count_only", "fact_quality_v1"]
    overall: Literal["weighted_average"]


class BenchmarkWeightingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta_fields: dict[str, dict[str, float]]
    aggregate: dict[str, float]

    @field_validator("meta_fields", mode="before")
    @classmethod
    def _validate_meta_fields(cls, value: Any) -> dict[str, dict[str, float]]:
        if not isinstance(value, dict):
            raise ValueError("weighting.meta_fields must be a mapping.")
        missing = [key for key in _ALLOWED_META_FIELDS if key not in value]
        extras = [str(key) for key in value.keys() if key not in _ALLOWED_META_FIELDS]
        if missing:
            raise ValueError("weighting.meta_fields is missing required keys: " + ", ".join(missing))
        if extras:
            raise ValueError("weighting.meta_fields contains unsupported keys: " + ", ".join(sorted(extras)))
        return {str(key): _normalize_meta_rule(rule) for key, rule in value.items()}

    @field_validator("aggregate", mode="before")
    @classmethod
    def _validate_aggregate(cls, value: Any) -> dict[str, float]:
        return _normalize_aggregate_weights(value)


class BenchmarkEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalize_inputs: bool = True
    prediction_format_default: Literal["auto", "page_level", "canonical_document"] = "auto"
    require_explicit_mappings: bool = True


class BenchmarkMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_file: Path
    gt_file: Path
    prediction_format: Literal["auto", "page_level", "canonical_document"] | None = None
    gt_page_index: int | None = Field(default=None, ge=0)

    @field_validator("prediction_file")
    @classmethod
    def _validate_prediction_file(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("prediction_file must be relative to benchmark.input_dir.")
        if str(path).strip() in {"", "."}:
            raise ValueError("prediction_file must not be empty.")
        return path

    @field_validator("gt_file")
    @classmethod
    def _validate_gt_file(cls, value: Path) -> Path:
        path = Path(value)
        if str(path).strip() in {"", "."}:
            raise ValueError("gt_file must not be empty.")
        return path

    @model_validator(mode="after")
    def _validate_page_level_index(self) -> "BenchmarkMappingConfig":
        if self.prediction_format == "page_level" and self.gt_page_index is None:
            raise ValueError("gt_page_index is required when prediction_format is page_level.")
        return self


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: BenchmarkPathsConfig
    methods: BenchmarkMethodsConfig
    weighting: BenchmarkWeightingConfig
    evaluation: BenchmarkEvaluationConfig
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    mappings: list[BenchmarkMappingConfig]

    @field_validator("model_metadata", mode="before")
    @classmethod
    def _validate_model_metadata(cls, value: Any) -> dict[str, Any]:
        return normalize_submission_defaults_partial(value)

    @model_validator(mode="after")
    def _validate_mappings(self) -> "BenchmarkConfig":
        if self.evaluation.require_explicit_mappings and not self.mappings:
            raise ValueError("mappings must be provided when evaluation.require_explicit_mappings is true.")
        self.benchmark.input_dir = self.benchmark.input_dir.expanduser()
        self.benchmark.output_dir = self.benchmark.output_dir.expanduser()
        return self


def _read_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read benchmark config files. Install with `pip install pyyaml`.") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML object at top level: {path}")
    return raw


def load_benchmark_config(config_path: Path | str) -> BenchmarkConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    payload = _read_yaml_file(path)
    try:
        return BenchmarkConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid benchmark config at {path}:\n{exc}") from exc
