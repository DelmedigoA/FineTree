from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .models import (
    BenchmarkGlobalAggregationSpec,
    BenchmarkNormalizeConfig,
    BenchmarkReportSpec,
    EvaluatorFieldSpec,
    EvaluatorSectionSpec,
    EvaluatorSpec,
    MetricGroupSpec,
    MetaFieldSpec,
)


try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised via runtime fallback
    yaml = None


def _load_yaml_payload(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        text = resources.files("benchmark_new").joinpath("configs/evaluation_specs.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Evaluation specs YAML must decode to an object.")
    return payload


def _coerce_global_aggregation(raw: dict[str, Any]) -> BenchmarkGlobalAggregationSpec:
    method = str(raw.get("method") or "").strip()
    if method != "weighted_mean":
        raise ValueError(f"Unsupported global aggregation method: {method}")
    return BenchmarkGlobalAggregationSpec(
        method=method,
        normalize_weights=bool(raw.get("normalize_weights", True)),
    )


def _coerce_report(raw: dict[str, Any]) -> BenchmarkReportSpec:
    return BenchmarkReportSpec(
        per_field=bool(raw.get("per_field", True)),
        per_document=bool(raw.get("per_document", False)),
        include_std=bool(raw.get("include_std", False)),
    )


def _coerce_normalize_config(raw: dict[str, Any]) -> BenchmarkNormalizeConfig:
    return BenchmarkNormalizeConfig(
        strip=bool(raw.get("strip", True)),
        quotes_unify=bool(raw.get("quotes_unify", True)),
        lowercase=bool(raw.get("lowercase", False)),
    )


def _comparison_type(compare_method: str) -> str:
    mapping = {
        "string_similarity": "soft",
        "exact_match": "hard",
        "path_all_levels_threshold": "hard",
        "path_last_leaf_threshold": "hard",
        "path_soft_overlap": "soft",
    }
    if compare_method not in mapping:
        raise ValueError(f"Unsupported compare_method: {compare_method}")
    return mapping[compare_method]


def _summary_metric(metrics: tuple[str, ...]) -> str:
    if not metrics:
        raise ValueError("Field config must declare at least one metric.")
    return metrics[0]


def _aggregation_label(averaging_rule: str, metrics: tuple[str, ...]) -> str:
    return f"{averaging_rule}/{'+'.join(metrics)}"


def _coerce_metric_groups(key: str, raw: Any) -> dict[str, MetricGroupSpec]:
    if not isinstance(raw, dict):
        raise TypeError(f"Field={key} metrics must be either a list or an object of metric groups.")
    metric_groups: dict[str, MetricGroupSpec] = {}
    for group_name, group_payload in raw.items():
        on: str | None = None
        if isinstance(group_payload, list):
            metrics_source = group_payload
        elif isinstance(group_payload, dict):
            metrics_source = group_payload.get("metrics")
            on = None if group_payload.get("on") in (None, "") else str(group_payload.get("on")).strip()
            if not isinstance(metrics_source, list):
                raise TypeError(f"Field={key} metric group={group_name} must declare a metrics list.")
        else:
            raise TypeError(f"Field={key} metric group={group_name} must be a list or object.")
        cleaned = tuple(str(metric).strip() for metric in metrics_source if str(metric).strip())
        if not cleaned:
            raise ValueError(f"Field={key} metric group={group_name} must contain at least one metric.")
        metric_groups[str(group_name)] = MetricGroupSpec(metrics=cleaned, on=on)
    return metric_groups


def _coerce_field(key: str, raw: dict[str, Any]) -> EvaluatorFieldSpec:
    parts = key.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Field key must be namespaced like section.field: {key}")
    section_name, field_name = parts
    if key == "facts.values":
        raise ValueError("Use `facts.value`, not `facts.values`.")
    compare_methods_raw = raw.get("compare_methods")
    compare_methods: tuple[str, ...] = ()
    if compare_methods_raw not in (None, ""):
        if not isinstance(compare_methods_raw, list):
            raise TypeError(f"Field={key} compare_methods must be a list.")
        compare_methods = tuple(str(item).strip() for item in compare_methods_raw if str(item).strip())
        if not compare_methods:
            raise ValueError(f"Field={key} compare_methods must contain at least one method.")
    compare_method = str(raw.get("compare_method") or (compare_methods[0] if compare_methods else "")).strip()
    averaging_rule = str(raw.get("averaging_rule") or "").strip()
    aggregate_on = None if raw.get("aggregate_on") in (None, "") else str(raw.get("aggregate_on")).strip()
    null_handling = str(raw.get("null_handling") or "regular").strip()
    metrics_raw = raw.get("metrics")
    metric_groups: dict[str, MetricGroupSpec] = {}
    if isinstance(metrics_raw, list):
        metrics = tuple(str(metric).strip() for metric in metrics_raw if str(metric).strip())
    else:
        metric_groups = _coerce_metric_groups(key, metrics_raw)
        metrics = tuple(metric for group in metric_groups.values() for metric in group.metrics)
    weight = float(raw.get("weight", 1.0))
    matcher = None if raw.get("matcher") in (None, "") else str(raw.get("matcher"))
    alignment_on_raw = raw.get("alignment_on") or []
    if not isinstance(alignment_on_raw, list):
        raise TypeError(f"Field={key} alignment_on must be a list.")
    alignment_on = tuple(str(item).strip() for item in alignment_on_raw if str(item).strip())
    dash_normalizer = None if raw.get("dash_normalizer") in (None, "") else str(raw.get("dash_normalizer"))
    special_value_normalization = {
        str(source): str(target)
        for source, target in dict(raw.get("special_value_normalization") or {}).items()
    }
    return EvaluatorFieldSpec(
        key=key,
        field=field_name,
        label=field_name,
        section=section_name,
        compare_method=compare_method,
        averaging_rule=averaging_rule,
        aggregate_on=aggregate_on,
        normalize=bool(raw.get("normalize", False)),
        null_handling=null_handling,
        comparison_type=_comparison_type(compare_method),
        aggregation=_aggregation_label(averaging_rule, metrics),
        summary_metric=_summary_metric(metrics),
        compare_methods=compare_methods,
        metrics=metrics,
        metric_groups=metric_groups,
        matcher=matcher,
        alignment_on=alignment_on,
        threshold=None if raw.get("threshold") in (None, "") else float(raw.get("threshold")),
        require_same_length=None if raw.get("require_same_length") in (None, "") else bool(raw.get("require_same_length")),
        dash_normalizer=dash_normalizer,
        special_value_normalization=special_value_normalization,
        weight=weight,
    )


def _build_page_meta_spec(payload: dict[str, Any]) -> EvaluatorSpec:
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        raise TypeError("Evaluation specs YAML must declare a 'fields' object.")
    grouped_fields: dict[str, list[EvaluatorFieldSpec]] = defaultdict(list)
    for key, value in raw_fields.items():
        if not isinstance(value, dict):
            continue
        field = _coerce_field(str(key), value)
        if field.section == "meta":
            grouped_fields[field.section].append(field)
    if "meta" not in grouped_fields:
        raise ValueError("Evaluation specs YAML must include at least one meta.* field.")
    sections = tuple(
        EvaluatorSectionSpec(name=section_name, label=section_name.replace("_", " ").title(), fields=tuple(fields))
        for section_name, fields in grouped_fields.items()
    )
    return EvaluatorSpec(
        name="page_meta",
        label="Page Meta",
        benchmark_version=None if payload.get("benchmark_version") in (None, "") else str(payload.get("benchmark_version")),
        global_aggregation=_coerce_global_aggregation(dict(payload.get("global_aggregation") or {})),
        report=_coerce_report(dict(payload.get("report") or {})),
        normalize_config=_coerce_normalize_config(dict(payload.get("normalize_config") or {})),
        output_file=None if payload.get("output_file") in (None, "") else str(payload.get("output_file")),
        sections=sections,
    )


def _build_facts_spec(payload: dict[str, Any], page_meta: EvaluatorSpec) -> EvaluatorSpec:
    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, dict):
        raise TypeError("Evaluation specs YAML must declare a 'fields' object.")
    grouped_fields: dict[str, list[EvaluatorFieldSpec]] = defaultdict(list)
    for key, value in raw_fields.items():
        if not isinstance(value, dict):
            continue
        field = _coerce_field(str(key), value)
        if field.section == "facts":
            grouped_fields[field.section].append(field)
    messages = dict(payload.get("under_development_messages") or {})
    sections = tuple(
        EvaluatorSectionSpec(name=section_name, label=section_name.replace("_", " ").title(), fields=tuple(fields))
        for section_name, fields in grouped_fields.items()
    )
    return EvaluatorSpec(
        name="facts",
        label="Facts",
        benchmark_version=page_meta.benchmark_version,
        global_aggregation=page_meta.global_aggregation,
        report=page_meta.report,
        normalize_config=page_meta.normalize_config,
        under_development_message=str(messages.get("facts") or "Facts evaluation is under development."),
        sections=sections,
    )


def load_evaluation_specs_from_path(path: Path | None = None) -> dict[str, EvaluatorSpec]:
    payload = _load_yaml_payload(path)
    page_meta = _build_page_meta_spec(payload)
    return {
        "page_meta": page_meta,
        "facts": _build_facts_spec(payload, page_meta),
    }


@lru_cache(maxsize=1)
def load_evaluation_specs() -> dict[str, EvaluatorSpec]:
    return load_evaluation_specs_from_path()


def get_evaluator_spec(name: str) -> EvaluatorSpec:
    specs = load_evaluation_specs()
    if name not in specs:
        raise KeyError(f"Unknown evaluator spec: {name}")
    return specs[name]


def get_page_meta_field_specs() -> tuple[MetaFieldSpec, ...]:
    evaluator = get_evaluator_spec("page_meta")
    return tuple(
        MetaFieldSpec(field=field.field, comparison_type=field.comparison_type)
        for section in evaluator.sections
        for field in section.fields
    )


def get_page_meta_normalized_fields() -> tuple[str, ...]:
    evaluator = get_evaluator_spec("page_meta")
    return tuple(
        field.field
        for section in evaluator.sections
        for field in section.fields
        if field.normalize
    )


def get_page_meta_normalize_config() -> BenchmarkNormalizeConfig:
    evaluator = get_evaluator_spec("page_meta")
    if evaluator.normalize_config is None:
        return BenchmarkNormalizeConfig(strip=True, quotes_unify=True, lowercase=False)
    return evaluator.normalize_config


def get_facts_field_spec(field_name: str) -> EvaluatorFieldSpec | None:
    evaluator = get_evaluator_spec("facts")
    for section in evaluator.sections:
        for field in section.fields:
            if field.field == field_name:
                return field
    return None
