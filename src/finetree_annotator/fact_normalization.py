from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .date_normalization import normalize_date
from .schema_contract import CANONICAL_FACT_KEYS as _CANONICAL_FACT_KEY_TUPLE
from .schema_contract import CURRENCY_VALUES, LEGACY_FACT_KEYS as _LEGACY_FACT_KEY_TUPLE, SCALE_VALUES, VALUE_TYPE_VALUES

CANONICAL_FACT_KEYS = set(_CANONICAL_FACT_KEY_TUPLE)
LEGACY_FACT_KEYS = set(_LEGACY_FACT_KEY_TUPLE)
CANONICAL_AND_META_KEYS = CANONICAL_FACT_KEYS | {"bbox"}
_VALID_CURRENCIES = set(CURRENCY_VALUES)
_VALID_SCALES = set(SCALE_VALUES)
_VALID_VALUE_TYPES = set(VALUE_TYPE_VALUES)
_NUMERIC_OR_PAREN_RE = re.compile(r"^\(?\d+(?:\.\d+)?\)?$")
_NEGATIVE_RE = re.compile(r"^-(\d+(?:\.\d+)?)$")
_RANGE_VALUE_RE = re.compile(r"^\d+\s*-\s*\d+$")
_SINGLE_ALLOWED_DASH = "-"
_EMPTY_DASH_PLACEHOLDERS = {"—", "–"}
_CURRENCY_SIGNS_RE = re.compile(r"[$₪€£]")
_CURRENCY_CODES_RE = re.compile(r"\b(?:USD|ILS|EUR|GBP)\b", flags=re.IGNORECASE)
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
        return "%"
    if lowered in {"amount", "regular"}:
        return "amount"
    return text if text in _VALID_VALUE_TYPES else None


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
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return "", []

    if "%" in raw_text:
        # User explicitly requested permissive handling for values containing '%'.
        return raw_text, []

    if raw_text == _SINGLE_ALLOWED_DASH:
        return _SINGLE_ALLOWED_DASH, []

    if raw_text in _EMPTY_DASH_PLACEHOLDERS:
        return "", ["placeholder_value"]

    # Some source pages prefix numeric values with '*' as a marker.
    # Strip leading marker(s) before numeric normalization.
    raw_text = re.sub(r"^\*+\s*", "", raw_text)

    cleaned = _CURRENCY_SIGNS_RE.sub("", raw_text)
    cleaned = _CURRENCY_CODES_RE.sub("", cleaned)
    cleaned = cleaned.replace(",", "")
    cleaned = "".join(cleaned.split())

    # After stripping currency tokens, placeholders like "$ -" become "-".
    if cleaned == _SINGLE_ALLOWED_DASH:
        return _SINGLE_ALLOWED_DASH, []
    if cleaned in _EMPTY_DASH_PLACEHOLDERS:
        return "", []

    neg_match = _NEGATIVE_RE.match(cleaned)
    if neg_match:
        cleaned = f"({neg_match.group(1)})"

    if _NUMERIC_OR_PAREN_RE.match(cleaned):
        return cleaned, []

    return raw_text, ["noncanonical_value"]


def _has_canonical_markers(raw_fact: Mapping[str, Any]) -> bool:
    return any(
        key in raw_fact
        for key in ("ref_comment", "comment", "note_flag", "is_note", "note_name", "ref_note", "note_ref", "note_reference", "note_num")
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

    ref_note_raw = payload.get("ref_note")
    if ref_note_raw in ("", None):
        ref_note_raw = payload.get("note_ref", payload.get("note_reference"))
    if ref_note_raw in ("", None):
        ref_note_raw = payload.get("refference", payload.get("reference", payload.get("ref")))
    ref_note = _to_optional_text(ref_note_raw)

    normalized_value_type = _normalize_value_type(payload.get("value_type"))
    raw_value_text = str(payload.get("value") or "").strip()
    value_input: Any = payload.get("value")
    keep_percent_range_value = bool(
        raw_value_text and _RANGE_VALUE_RE.match(raw_value_text) and normalized_value_type == "%"
    )
    if raw_value_text and _RANGE_VALUE_RE.match(raw_value_text) and normalized_value_type != "%":
        if not ref_note:
            ref_note = raw_value_text.replace(" ", "")
        value_input = ""

    note_flag_raw = payload.get("note_flag", payload.get("is_note"))
    if note_flag_raw in ("", None):
        note_flag_raw = payload.get("is_beur", payload.get("beur"))
    note_flag, bool_warnings = _coerce_note_flag(note_flag_raw)
    note_num, _note_num_warnings = normalize_note_num(note_num_raw)

    if keep_percent_range_value:
        value, value_warnings = raw_value_text, []
    else:
        value, value_warnings = normalize_value(value_input)
    date_value, date_warnings = normalize_date(payload.get("date"))

    normalized: dict[str, Any] = {
        "value": value,
        "ref_comment": _to_optional_text(comment_raw),
        "note_flag": note_flag,
        "note_name": _to_optional_text(note_name_raw),
        "note_num": note_num,
        "ref_note": ref_note,
        "date": date_value,
        "path": _normalize_path(payload.get("path")),
        "currency": _normalize_currency(payload.get("currency")),
        "scale": _normalize_scale(payload.get("scale")),
        "value_type": normalized_value_type,
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
    findings: list[dict[str, Any]] = []
    pages = out.get("pages")
    if isinstance(pages, list):
        new_pages: list[dict[str, Any]] = []
        for page_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_out = dict(page)
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
