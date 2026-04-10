from __future__ import annotations

import re
from datetime import date
from typing import Any

from .evaluation_specs import get_facts_field_spec, get_page_meta_field_specs, get_page_meta_normalize_config, get_page_meta_normalized_fields
from .models import FactFieldSpec, MetaFieldSpec


META_FIELDS: tuple[MetaFieldSpec, ...] = get_page_meta_field_specs()
META_NORMALIZED_FIELDS: tuple[str, ...] = get_page_meta_normalized_fields()
META_NORMALIZE_CONFIG = get_page_meta_normalize_config()
FACT_VALUE_FIELD_SPEC = get_facts_field_spec("value")
FACT_PATH_FIELD_SPEC = get_facts_field_spec("path")

FACT_FIELDS: tuple[FactFieldSpec, ...] = (
    FactFieldSpec(field="value", primary=True, score_formula="mean(exact_f1, numeric_closeness_score)"),
    FactFieldSpec(field="path", primary=True, score_formula="accuracy"),
    FactFieldSpec(field="fact_num", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="period_type", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="period_start", primary=True, score_formula="mean(exact_f1, date_diff_score)"),
    FactFieldSpec(field="period_end", primary=True, score_formula="mean(exact_f1, date_diff_score)"),
    FactFieldSpec(field="currency", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="scale", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="value_type", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="value_context", primary=True, score_formula="exact_f1"),
    FactFieldSpec(field="comment_ref", primary=False, score_formula="exact_metrics + string_similarity"),
    FactFieldSpec(field="note_name", primary=False, score_formula="exact_metrics + string_similarity"),
    FactFieldSpec(field="note_flag", primary=False, score_formula="exact_f1"),
    FactFieldSpec(field="note_num", primary=False, score_formula="exact_f1"),
    FactFieldSpec(field="note_ref", primary=False, score_formula="exact_f1"),
    FactFieldSpec(field="path_source", primary=False, score_formula="exact_f1"),
)

PRIMARY_FACT_FIELD_NAMES: tuple[str, ...] = tuple(field.field for field in FACT_FIELDS if field.primary)
DATE_FACT_FIELDS: tuple[str, ...] = ("period_start", "period_end")
SOFT_STRING_FACT_FIELDS: tuple[str, ...] = ("comment_ref", "note_name")
SOFT_META_FIELDS: tuple[str, ...] = tuple(field.field for field in META_FIELDS if field.comparison_type == "soft")
SPARSE_OPTIONAL_FACT_FIELDS: tuple[str, ...] = ("comment_ref", "note_name", "note_flag", "note_num", "note_ref", "path_source")
_NUMERIC_PATTERN = re.compile(r"[^0-9().,+-]")
_SOFT_QUOTE_TRANSLATION = str.maketrans(
    {
        "״": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "׳": "'",
        "‘": "'",
        "’": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")
_GROUPING_SEPARATOR_RE = re.compile(r"(?<=\d)[,\s_](?=\d)")
_VALUE_FOOTNOTE_TOKEN_RE = re.compile(r"\(\*\)|\*")


def _normalize_value_string(value: Any) -> str | None:
    text = normalize_scalar(value)
    if text is None:
        return None
    if FACT_VALUE_FIELD_SPEC is not None and not FACT_VALUE_FIELD_SPEC.normalize:
        return str(text)
    normalized = str(text).strip()
    normalized = _VALUE_FOOTNOTE_TOKEN_RE.sub("", normalized)
    normalized = _WHITESPACE_RE.sub("", normalized)
    normalized = _GROUPING_SEPARATOR_RE.sub("", normalized)
    normalized = normalized.strip()
    if FACT_VALUE_FIELD_SPEC is not None and normalized in FACT_VALUE_FIELD_SPEC.special_value_normalization:
        return FACT_VALUE_FIELD_SPEC.special_value_normalization[normalized]
    if FACT_VALUE_FIELD_SPEC is not None and FACT_VALUE_FIELD_SPEC.dash_normalizer == "treat_as_zero" and normalized == "-":
        return "0"
    if not normalized:
        return None
    if normalized.startswith("-"):
        stripped = normalized[1:].strip()
        if not stripped:
            return "0" if FACT_VALUE_FIELD_SPEC is not None and FACT_VALUE_FIELD_SPEC.dash_normalizer == "treat_as_zero" else None
        return f"({stripped})"
    return normalized


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    return text if text else None


def normalize_soft_string(value: Any) -> str | None:
    text = normalize_scalar(value)
    if text is None:
        return None
    normalized = str(text)
    if META_NORMALIZE_CONFIG.quotes_unify:
        normalized = normalized.translate(_SOFT_QUOTE_TRANSLATION)
    if META_NORMALIZE_CONFIG.strip:
        normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    if META_NORMALIZE_CONFIG.lowercase:
        normalized = normalized.lower()
    return normalized or None


def normalize_meta_value(field: str, value: Any) -> Any:
    if field in SOFT_META_FIELDS and field in META_NORMALIZED_FIELDS:
        return normalize_soft_string(value)
    if field in SOFT_META_FIELDS:
        return normalize_scalar(value)
    return normalize_scalar(value)


def normalize_fact_value(field: str, value: Any) -> Any:
    if field == "path":
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if FACT_PATH_FIELD_SPEC is not None and FACT_PATH_FIELD_SPEC.normalize:
                normalized_item = normalize_soft_string(item)
            else:
                normalized_item = normalize_scalar(item)
            if normalized_item is None:
                continue
            normalized.append(str(normalized_item))
        return normalized
    if field == "value":
        return _normalize_value_string(value)
    if field in SOFT_STRING_FACT_FIELDS:
        return normalize_soft_string(value)
    return normalize_scalar(value)


def joined_path(path: Any) -> str:
    if not isinstance(path, list):
        return ""
    return " / ".join(str(item).strip() for item in path if str(item).strip())


def parse_numeric_value(value: Any) -> float | None:
    text = _normalize_value_string(value)
    if text is None:
        return None
    raw = _NUMERIC_PATTERN.sub("", str(text))
    if not raw:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.replace(",", "").replace("(", "").replace(")", "")
    if cleaned in {"", "-", "+", ".", "-.", "+."}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return -parsed if negative else parsed


def parse_date_value(value: Any) -> date | None:
    text = normalize_scalar(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(str(text))
    except ValueError:
        return None


def primary_fact_fields() -> list[FactFieldSpec]:
    return [field for field in FACT_FIELDS if field.primary]
