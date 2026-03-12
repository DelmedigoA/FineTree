from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .date_normalization import normalize_date
from .schema_contract import CANONICAL_FACT_KEYS as _CANONICAL_FACT_KEY_TUPLE
from .schema_contract import (
    CURRENCY_VALUES,
    LEGACY_FACT_KEYS as _LEGACY_FACT_KEY_TUPLE,
    SCALE_VALUES,
    VALUE_TYPE_VALUES,
)
from .schemas import is_legacy_page_type_value, split_legacy_page_type

CANONICAL_FACT_KEYS = set(_CANONICAL_FACT_KEY_TUPLE)
LEGACY_FACT_KEYS = set(_LEGACY_FACT_KEY_TUPLE) | {"balance_type", "aggregation_role"}
CANONICAL_AND_META_KEYS = CANONICAL_FACT_KEYS | {"bbox"}
_VALID_CURRENCIES = set(CURRENCY_VALUES)
_VALID_SCALES = set(SCALE_VALUES)
_VALID_VALUE_TYPES = set(VALUE_TYPE_VALUES)
_DATE_YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_YEAR_RE = re.compile(r"^\d{4}$")
_SINGLE_ALLOWED_DASH = "-"
_EMPTY_DASH_PLACEHOLDERS = {"—", "–"}
_ROW_TOTAL_MARKERS: tuple[str, ...] = (
    'סה"כ',
    "סה״כ",
    "סהכ",
    "סך",
    "סך הכל",
    "total",
    "subtotal",
    "net subtotal",
    "net total",
    "net",
)
DEFAULT_ANNOTATIONS_GLOB = "data/annotations/*.json"


def _to_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_path(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _normalize_currency(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    upper = text.upper()
    return upper if upper in _VALID_CURRENCIES else None


def _normalize_scale(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed in _VALID_SCALES else None


def _normalize_value_type(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"%", "percent", "percentage"}:
        return "percent"
    if lowered in {"amount", "regular"}:
        return "amount"
    if lowered in {"ratio", "count"}:
        return lowered
    return text if text in _VALID_VALUE_TYPES else None


def _normalize_value_context(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if lowered in {"textual", "tabular", "mixed"} else None


def _normalize_row_role(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    aliases = {
        "detail": "detail",
        "details": "detail",
        "child": "detail",
        "line": "detail",
        "total": "total",
        "subtotal": "total",
        "net": "total",
        "summary": "total",
    }
    normalized = aliases.get(lowered)
    if normalized is not None:
        return normalized
    return None


def _normalize_text_for_role_inference(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = (
        text.replace("״", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("׳", "'")
        .replace("–", "-")
        .replace("—", "-")
    )
    return re.sub(r"\s+", " ", text)
def _infer_row_role(
    *,
    path: list[str],
    comment_ref: Any,
    note_name: Any,
    note_ref: Any,
) -> str:
    contexts: list[str] = []
    for candidate in (comment_ref, note_name, note_ref):
        normalized = _normalize_text_for_role_inference(candidate)
        if normalized:
            contexts.append(normalized)
    for level in path:
        normalized = _normalize_text_for_role_inference(level)
        if normalized:
            contexts.append(normalized)
    for text in contexts:
        if any(marker in text for marker in _ROW_TOTAL_MARKERS):
            return "total"
    return "detail"


def _derive_natural_sign_from_value(value: str) -> str | None:
    text = str(value or "").strip()
    if text == "-":
        return None
    if "(" in text and ")" in text:
        return "negative"
    return "positive"


def _normalize_equation(value: Any) -> str | None:
    return _to_optional_text(value)


def _normalize_legacy_equation_children(value: Any) -> list[dict[str, Any]] | None:
    if value in ("", None):
        return None
    if not isinstance(value, list):
        return None
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        fact_num = _normalize_fact_num(entry.get("fact_num"))
        operator = str(entry.get("operator") or "").strip()
        if fact_num is None or operator not in {"+", "-"}:
            continue
        out.append({"fact_num": fact_num, "operator": operator})
    return out or None


def _fact_equation_from_children(value: Any) -> str | None:
    children = _normalize_legacy_equation_children(value)
    if not children:
        return None
    terms: list[str] = []
    for idx, child in enumerate(children):
        fact_num = int(child["fact_num"])
        operator = str(child.get("operator") or "+")
        if idx == 0:
            terms.append(f"- f{fact_num}" if operator == "-" else f"f{fact_num}")
        else:
            terms.append(f"{operator} f{fact_num}")
    text = " ".join(terms).strip()
    return text or None


def _equation_signature(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    equation = str(entry.get("equation") or "")
    fact_equation = str(entry.get("fact_equation") or "")
    return equation, fact_equation


def _normalize_equation_entry(
    value: Any,
    *,
    allow_equation_only_string: bool = True,
) -> dict[str, Any] | None:
    if isinstance(value, str):
        if not allow_equation_only_string:
            return None
        equation = _normalize_equation(value)
        if equation is None:
            return None
        return {"equation": equation, "fact_equation": None}
    if not isinstance(value, Mapping):
        return None
    equation = _normalize_equation(value.get("equation"))
    if equation is None:
        return None
    return {
        "equation": equation,
        "fact_equation": _normalize_equation(value.get("fact_equation")),
    }


def _normalize_equations(value: Any) -> list[dict[str, Any]] | None:
    if value in ("", None):
        return None
    if not isinstance(value, list):
        return None
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for entry in value:
        normalized_entry = _normalize_equation_entry(entry)
        if normalized_entry is None:
            continue
        signature = _equation_signature(normalized_entry)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(normalized_entry)
    return out or None


def _normalize_fact_num(value: Any) -> int | None:
    if value in ("", None):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float) and float(value).is_integer():
        parsed = int(value)
        return parsed if parsed >= 1 else None
    text = str(value).strip()
    if text.isdigit():
        parsed = int(text)
        return parsed if parsed >= 1 else None
    return None


def _normalize_period_type(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if lowered in {"instant", "duration", "expected"} else None


def _normalize_duration_type(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"recurring", "recurrent"}:
        return "recurrent"
    return None


def _normalize_recurring_period(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if lowered in {"daily", "quarterly", "monthly", "yearly"} else None


def _normalize_day_date(value: Any) -> str | None:
    normalized, warnings = normalize_date(value)
    if warnings or normalized is None or len(normalized) != 10:
        return None
    return normalized


def _infer_period_from_date(
    date_value: str | None,
    *,
    period_type: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not date_value:
        return None, None, None

    if period_type == "duration":
        if _DATE_YEAR_RE.fullmatch(date_value) or _DATE_YMD_RE.fullmatch(date_value):
            year = date_value[:4]
            return "duration", f"{year}-01-01", f"{year}-12-31"
        return None, None, None

    if period_type == "instant":
        if _DATE_YMD_RE.fullmatch(date_value):
            return "instant", None, date_value
        return None, None, None

    if _DATE_YMD_RE.fullmatch(date_value):
        return "instant", None, date_value
    if _DATE_YEAR_RE.fullmatch(date_value):
        return "duration", f"{date_value}-01-01", f"{date_value}-12-31"
    return None, None, None


def _normalize_path_source(value: Any) -> str | None:
    text = _to_optional_text(value)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if lowered in {"observed", "inferred"} else None


def normalize_note_num(raw_note_num: Any) -> tuple[int | None, list[str]]:
    if raw_note_num in ("", None):
        return None, []
    if isinstance(raw_note_num, bool):
        return None, ["noninteger_note_num"]
    if isinstance(raw_note_num, int):
        return raw_note_num, []
    if isinstance(raw_note_num, float) and float(raw_note_num).is_integer():
        return int(raw_note_num), []

    text = str(raw_note_num).strip()
    if not text:
        return None, []
    if text.isdigit():
        return int(text), []
    return None, ["noninteger_note_num"]


def _coerce_note_flag(value: Any) -> tuple[bool, list[str]]:
    if value in ("", None):
        return False, []
    if isinstance(value, bool):
        return value, []
    if isinstance(value, (int, float)):
        return bool(value), []
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True, []
    if lowered in {"false", "0", "no", "n"}:
        return False, []
    return False, ["nonboolean_note_flag"]

def normalize_value(raw_value: Any) -> tuple[str, list[str]]:
    if raw_value is None:
        return _SINGLE_ALLOWED_DASH, ["empty_value"]
    raw_text = str(raw_value).strip()
    if not raw_text:
        return _SINGLE_ALLOWED_DASH, ["empty_value"]
    if raw_text in _EMPTY_DASH_PLACEHOLDERS:
        return _SINGLE_ALLOWED_DASH, ["placeholder_value"]
    return raw_text, []


def _has_canonical_markers(raw_fact: Mapping[str, Any]) -> bool:
    return any(
        key in raw_fact
        for key in (
            "comment_ref",
            "ref_comment",
            "comment",
            "note_flag",
            "is_note",
            "note_name",
            "note_ref",
            "ref_note",
            "note_reference",
            "note_num",
            "fact_num",
            "fact_equation",
            "equations",
            "balance_type",
            "natural_sign",
            "row_role",
            "aggregation_role",
            "period_type",
            "period_start",
            "period_end",
            "duration_type",
            "recurring_period",
            "value_context",
            "path_source",
        )
    )


def _has_legacy_markers(raw_fact: Mapping[str, Any]) -> bool:
    return any(key in raw_fact for key in LEGACY_FACT_KEYS)


def normalize_fact_payload(
    raw_fact: Mapping[str, Any] | None,
    *,
    include_bbox: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    payload = raw_fact if isinstance(raw_fact, Mapping) else {}
    has_canonical = _has_canonical_markers(payload)
    has_legacy = _has_legacy_markers(payload)

    if has_canonical:
        comment_raw = payload.get("comment_ref")
        if comment_raw in ("", None):
            comment_raw = payload.get("ref_comment")
        if comment_raw in ("", None):
            comment_raw = payload.get("comment")
        if comment_raw in ("", None) and has_legacy:
            comment_raw = payload.get("note")
        note_name_raw = payload.get("note_name", payload.get("beur_name"))
        note_num_raw = payload.get("note_num", payload.get("note"))
        if note_num_raw in ("", None) and has_legacy:
            note_num_raw = payload.get("beur_num", payload.get("beur_number"))
    else:
        comment_raw = payload.get("note")
        note_name_raw = payload.get("beur_name")
        note_num_raw = payload.get("beur_num", payload.get("beur_number"))

    ref_note_raw = payload.get("note_ref")
    if ref_note_raw in ("", None):
        ref_note_raw = payload.get("ref_note")
    if ref_note_raw in ("", None):
        ref_note_raw = payload.get("note_reference")
    if ref_note_raw in ("", None):
        ref_note_raw = payload.get("refference", payload.get("reference", payload.get("ref")))
    note_ref = _to_optional_text(ref_note_raw)
    path_value = _normalize_path(payload.get("path"))

    normalized_value_type = _normalize_value_type(payload.get("value_type"))
    value_input: Any = payload.get("value")

    note_flag_raw = payload.get("note_flag", payload.get("is_note"))
    if note_flag_raw in ("", None):
        note_flag_raw = payload.get("is_beur", payload.get("beur"))
    note_flag, bool_warnings = _coerce_note_flag(note_flag_raw)
    note_num, _note_num_warnings = normalize_note_num(note_num_raw)

    value, value_warnings = normalize_value(value_input)
    natural_sign = _derive_natural_sign_from_value(value)
    date_value, date_warnings = normalize_date(payload.get("date"))

    has_period_type = "period_type" in payload
    has_period_start = "period_start" in payload
    has_period_end = "period_end" in payload
    period_type = _normalize_period_type(payload.get("period_type"))
    period_start = _normalize_day_date(payload.get("period_start"))
    period_end = _normalize_day_date(payload.get("period_end"))
    inferred_type, inferred_start, inferred_end = _infer_period_from_date(date_value, period_type=period_type)
    if not has_period_type and period_type is None and inferred_type is not None:
        period_type = inferred_type
    if not has_period_start and period_start is None and inferred_start is not None:
        period_start = inferred_start
    if not has_period_end and period_end is None and inferred_end is not None:
        period_end = inferred_end
    duration_type = _normalize_duration_type(payload.get("duration_type"))
    recurring_period = _normalize_recurring_period(payload.get("recurring_period"))
    value_context = _normalize_value_context(payload.get("value_context"))
    row_role = _normalize_row_role(payload.get("row_role"))
    raw_aggregation_role_text = str(payload.get("aggregation_role") or "").strip().lower()
    if row_role is None and raw_aggregation_role_text == "total":
        row_role = "total"
    if row_role is None:
        row_role = _infer_row_role(
            path=path_value,
            comment_ref=comment_raw,
            note_name=note_name_raw,
            note_ref=note_ref,
        )

    legacy_equation = _normalize_equation(payload.get("equation"))
    legacy_fact_equation = _normalize_equation(payload.get("fact_equation"))
    if legacy_fact_equation is None:
        legacy_fact_equation = _fact_equation_from_children(payload.get("equation_children"))
    active_equation = _normalize_equation_entry(
        {
            "equation": legacy_equation,
            "fact_equation": legacy_fact_equation,
        },
        allow_equation_only_string=False,
    )
    equations = _normalize_equations(payload.get("equations"))
    if active_equation is not None:
        if equations is None:
            equations = [active_equation]
        else:
            active_signature = _equation_signature(active_equation)
            matching_index = next(
                (idx for idx, entry in enumerate(equations) if _equation_signature(entry) == active_signature),
                None,
            )
            if matching_index is None:
                equations = [active_equation, *equations]
            elif matching_index != 0:
                equations = [equations[matching_index], *equations[:matching_index], *equations[matching_index + 1 :]]
    normalized: dict[str, Any] = {
        "value": value,
        "fact_num": _normalize_fact_num(payload.get("fact_num")),
        "equations": equations,
        "natural_sign": natural_sign,
        "row_role": row_role,
        "comment_ref": _to_optional_text(comment_raw),
        "note_flag": note_flag,
        "note_name": _to_optional_text(note_name_raw),
        "note_num": note_num,
        "note_ref": note_ref,
        "date": date_value,
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "duration_type": duration_type,
        "recurring_period": recurring_period,
        "path": path_value,
        "path_source": _normalize_path_source(payload.get("path_source")),
        "currency": _normalize_currency(payload.get("currency")),
        "scale": _normalize_scale(payload.get("scale")),
        "value_type": normalized_value_type,
        "value_context": value_context,
    }
    if include_bbox and "bbox" in payload:
        normalized["bbox"] = payload.get("bbox")

    # Non-integer note_num is treated as a live/page warning, not a format-audit failure.
    warnings = bool_warnings + date_warnings + value_warnings
    return normalized, warnings


def _iter_pages_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pages")
    if isinstance(pages, list):
        return [page for page in pages if isinstance(page, dict)]
    facts = payload.get("facts")
    if isinstance(facts, list):
        image = str(payload.get("image") or "page")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        return [{"image": image, "meta": meta, "facts": facts}]
    return []


def normalize_annotation_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}, []

    out = dict(payload)
    raw_metadata = out.get("metadata", out.get("document_meta"))
    if isinstance(raw_metadata, dict):
        out["metadata"] = dict(raw_metadata)
    out.pop("document_meta", None)
    findings: list[dict[str, Any]] = []
    pages = out.get("pages")
    if isinstance(pages, list):
        new_pages: list[dict[str, Any]] = []
        for page_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_out = dict(page)
            meta = page_out.get("meta")
            if isinstance(meta, dict):
                normalized_meta = dict(meta)
                legacy_type = meta.get("type")
                if is_legacy_page_type_value(legacy_type):
                    page_type, statement_type = split_legacy_page_type(legacy_type)
                    if "page_type" not in normalized_meta or not str(normalized_meta.get("page_type") or "").strip():
                        normalized_meta["page_type"] = page_type
                    if "statement_type" not in normalized_meta and statement_type is not None:
                        normalized_meta["statement_type"] = statement_type
                elif (
                    "page_type" not in normalized_meta or not str(normalized_meta.get("page_type") or "").strip()
                ) and legacy_type not in ("", None):
                    normalized_meta["page_type"] = legacy_type
                normalized_meta.pop("type", None)
                page_out["meta"] = normalized_meta

            facts = page.get("facts")
            if not isinstance(facts, list):
                new_pages.append(page_out)
                continue
            normalized_facts: list[dict[str, Any]] = []
            for fact_idx, raw_fact in enumerate(facts):
                if not isinstance(raw_fact, dict):
                    continue
                normalized_fact, fact_warnings = normalize_fact_payload(raw_fact, include_bbox=("bbox" in raw_fact))
                issue_codes: list[str] = []
                if any(key in raw_fact for key in LEGACY_FACT_KEYS):
                    issue_codes.append("legacy_keys")
                unknown_keys = sorted(
                    key for key in raw_fact.keys() if key not in CANONICAL_AND_META_KEYS and key not in LEGACY_FACT_KEYS
                )
                if unknown_keys:
                    issue_codes.append("unknown_keys")
                issue_codes.extend(fact_warnings)
                if raw_fact != normalized_fact:
                    issue_codes.append("rewrite_needed")
                if issue_codes:
                    findings.append(
                        {
                            "page_index": page_idx,
                            "page": str(page.get("image") or f"page_{page_idx + 1}"),
                            "fact_index": fact_idx,
                            "issue_codes": sorted(set(issue_codes)),
                            "warnings": sorted(set(fact_warnings)),
                            "legacy_keys": sorted(key for key in raw_fact.keys() if key in LEGACY_FACT_KEYS),
                            "unknown_keys": unknown_keys,
                            "raw_value": str(raw_fact.get("value") or ""),
                            "normalized_value": str(normalized_fact.get("value") or ""),
                            "raw_date": raw_fact.get("date"),
                            "normalized_date": normalized_fact.get("date"),
                        }
                    )
                normalized_facts.append(normalized_fact)
            page_out["facts"] = normalized_facts
            new_pages.append(page_out)
        out["pages"] = new_pages
        return out, findings

    # Single-page payload support for older shapes.
    facts = out.get("facts")
    if isinstance(facts, list):
        normalized_facts: list[dict[str, Any]] = []
        for fact_idx, raw_fact in enumerate(facts):
            if not isinstance(raw_fact, dict):
                continue
            normalized_fact, fact_warnings = normalize_fact_payload(raw_fact, include_bbox=("bbox" in raw_fact))
            issue_codes: list[str] = []
            if any(key in raw_fact for key in LEGACY_FACT_KEYS):
                issue_codes.append("legacy_keys")
            unknown_keys = sorted(
                key for key in raw_fact.keys() if key not in CANONICAL_AND_META_KEYS and key not in LEGACY_FACT_KEYS
            )
            if unknown_keys:
                issue_codes.append("unknown_keys")
            issue_codes.extend(fact_warnings)
            if raw_fact != normalized_fact:
                issue_codes.append("rewrite_needed")
            if issue_codes:
                findings.append(
                    {
                        "page_index": 0,
                        "page": str(out.get("image") or "page"),
                        "fact_index": fact_idx,
                        "issue_codes": sorted(set(issue_codes)),
                        "warnings": sorted(set(fact_warnings)),
                        "legacy_keys": sorted(key for key in raw_fact.keys() if key in LEGACY_FACT_KEYS),
                        "unknown_keys": unknown_keys,
                        "raw_value": str(raw_fact.get("value") or ""),
                        "normalized_value": str(normalized_fact.get("value") or ""),
                        "raw_date": raw_fact.get("date"),
                        "normalized_date": normalized_fact.get("date"),
                    }
                )
            normalized_facts.append(normalized_fact)
        out["facts"] = normalized_facts
    return out, findings


def _normalize_included_doc_ids(include_doc_ids: set[str] | None) -> set[str] | None:
    if include_doc_ids is None:
        return None
    return {doc_id.strip() for doc_id in include_doc_ids if str(doc_id).strip()}


def _resolve_annotation_files(
    root: Path,
    annotations_glob: str,
    *,
    include_doc_ids: set[str] | None = None,
) -> list[Path]:
    root = root.resolve()
    pattern = Path(annotations_glob).expanduser()
    search_pattern = str(pattern) if pattern.is_absolute() else str(root / annotations_glob)
    resolved = sorted(Path(path).resolve() for path in glob.glob(search_pattern, recursive=True))
    included = _normalize_included_doc_ids(include_doc_ids)
    return [
        path
        for path in resolved
        if path.is_file() and (included is None or path.stem in included)
    ]


def fact_format_report(
    root: Path,
    *,
    annotations_glob: str = DEFAULT_ANNOTATIONS_GLOB,
    include_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    files = _resolve_annotation_files(root, annotations_glob, include_doc_ids=include_doc_ids)
    findings: list[dict[str, Any]] = []
    pages_scanned = 0
    facts_scanned = 0
    facts_with_issues = 0

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        _, payload_findings = normalize_annotation_payload(payload)
        pages_scanned += len(_iter_pages_from_payload(payload))
        page_facts = 0
        for page in _iter_pages_from_payload(payload):
            facts = page.get("facts")
            if isinstance(facts, list):
                page_facts += len([fact for fact in facts if isinstance(fact, dict)])
        facts_scanned += page_facts

        for finding in payload_findings:
            facts_with_issues += 1
            findings.append(
                {
                    "file": str(file_path.relative_to(root)),
                    **finding,
                }
            )

    return {
        "annotations_glob": annotations_glob,
        "include_doc_ids": sorted(_normalize_included_doc_ids(include_doc_ids) or set()),
        "files_scanned": len(files),
        "pages_scanned": pages_scanned,
        "facts_scanned": facts_scanned,
        "facts_with_issues": facts_with_issues,
        "findings": findings,
    }


def assert_fact_format(
    root: Path,
    *,
    annotations_glob: str = DEFAULT_ANNOTATIONS_GLOB,
    fail_on_issues: bool = True,
    include_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    report = fact_format_report(root, annotations_glob=annotations_glob, include_doc_ids=include_doc_ids)
    summary = {
        "annotations_glob": report["annotations_glob"],
        "include_doc_ids": report["include_doc_ids"],
        "files_scanned": report["files_scanned"],
        "pages_scanned": report["pages_scanned"],
        "facts_scanned": report["facts_scanned"],
        "facts_with_issues": report["facts_with_issues"],
        "findings_preview": report["findings"][:5],
    }
    print("FACT_FORMAT_AUDIT:", json.dumps(summary, ensure_ascii=False))
    if fail_on_issues and int(report["facts_with_issues"]) > 0:
        raise RuntimeError(
            "Fact schema/format violations detected in annotation JSON. "
            f"facts_with_issues={report['facts_with_issues']}. "
            "Run scripts/check_fact_schema_format.py for details, or pass --allow-format-issues to bypass."
        )
    return report


__all__ = [
    "CANONICAL_FACT_KEYS",
    "DEFAULT_ANNOTATIONS_GLOB",
    "LEGACY_FACT_KEYS",
    "assert_fact_format",
    "fact_format_report",
    "normalize_annotation_payload",
    "normalize_date",
    "normalize_fact_payload",
    "normalize_note_num",
    "normalize_value",
]
