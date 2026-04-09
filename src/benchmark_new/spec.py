from __future__ import annotations

import re
from datetime import date
from typing import Any

from .models import FactFieldSpec, MetaFieldSpec


META_FIELDS: tuple[MetaFieldSpec, ...] = (
    MetaFieldSpec(field="entity_name", comparison_type="soft"),
    MetaFieldSpec(field="page_num", comparison_type="hard"),
    MetaFieldSpec(field="page_type", comparison_type="hard"),
    MetaFieldSpec(field="statement_type", comparison_type="hard"),
    MetaFieldSpec(field="title", comparison_type="soft"),
)

FACT_FIELDS: tuple[FactFieldSpec, ...] = (
    FactFieldSpec(field="value", primary=True, score_formula="mean(exact_f1, numeric_closeness_score)"),
    FactFieldSpec(field="path", primary=True, score_formula="mean(exact_f1, joined_path_similarity, elementwise_path_similarity)"),
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
    normalized = str(text).translate(_SOFT_QUOTE_TRANSLATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized or None


def normalize_meta_value(field: str, value: Any) -> Any:
    if field in SOFT_META_FIELDS:
        return normalize_soft_string(value)
    return normalize_scalar(value)


def normalize_fact_value(field: str, value: Any) -> Any:
    if field == "path":
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
    if field == "value":
        text = normalize_scalar(value)
        numeric = parse_numeric_value(text)
        if numeric is None:
            return text
        if float(numeric).is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:,.15g}"
    if field in SOFT_STRING_FACT_FIELDS:
        return normalize_soft_string(value)
    return normalize_scalar(value)


def joined_path(path: Any) -> str:
    if not isinstance(path, list):
        return ""
    return " / ".join(str(item).strip() for item in path if str(item).strip())


def parse_numeric_value(value: Any) -> float | None:
    text = normalize_scalar(value)
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
