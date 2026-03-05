from __future__ import annotations

import glob
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping

CANONICAL_FACT_KEYS = {
    "value",
    "comment",
    "is_note",
    "note",
    "note_reference",
    "date",
    "path",
    "currency",
    "scale",
    "value_type",
}
LEGACY_FACT_KEYS = {
    "note",
    "is_beur",
    "beur_num",
    "refference",
    "reference",
    "ref",
    "beur",
    "beur_number",
    "footnote",
}
CANONICAL_AND_META_KEYS = CANONICAL_FACT_KEYS | {"bbox"}
_VALID_CURRENCIES = {"ILS", "USD", "EUR", "GBP"}
_VALID_SCALES = {1, 1000, 1_000_000}
_VALID_VALUE_TYPES = {"amount", "%"}
_DATE_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_YMD_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")
_DATE_DMY_RE = re.compile(r"^(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})$")
_NUMERIC_OR_PAREN_RE = re.compile(r"^\(?\d+(?:\.\d+)?\)?$")
_NEGATIVE_RE = re.compile(r"^-(\d+(?:\.\d+)?)$")
_DASH_PLACEHOLDERS = {"-", "—", "–"}
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


def _coerce_is_note(value: Any) -> tuple[bool, list[str]]:
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
    return False, ["nonboolean_is_note"]


def normalize_date(raw_date: Any) -> tuple[str | None, list[str]]:
    text = _to_optional_text(raw_date)
    if text is None:
        return None, []

    if _DATE_YEAR_RE.match(text):
        return text, []

    m_ymd = _DATE_YMD_RE.match(text)
    if m_ymd:
        try:
            date(
                int(m_ymd.group("year")),
                int(m_ymd.group("month")),
                int(m_ymd.group("day")),
            )
        except ValueError:
            return text, ["noncanonical_date"]
        return text, []

    m_dmy = _DATE_DMY_RE.match(text)
    if m_dmy:
        try:
            parsed = date(
                int(m_dmy.group("year")),
                int(m_dmy.group("month")),
                int(m_dmy.group("day")),
            )
        except ValueError:
            return text, ["noncanonical_date"]
        return parsed.strftime("%Y-%m-%d"), []

    return text, ["noncanonical_date"]


def normalize_value(raw_value: Any) -> tuple[str, list[str]]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return "", []

    if "%" in raw_text:
        # User explicitly requested permissive handling for values containing '%'.
        return raw_text, []

    if raw_text in _DASH_PLACEHOLDERS:
        return "", ["placeholder_value"]

    cleaned = _CURRENCY_SIGNS_RE.sub("", raw_text)
    cleaned = _CURRENCY_CODES_RE.sub("", cleaned)
    cleaned = cleaned.replace(",", "")
    cleaned = "".join(cleaned.split())

    neg_match = _NEGATIVE_RE.match(cleaned)
    if neg_match:
        cleaned = f"({neg_match.group(1)})"

    if _NUMERIC_OR_PAREN_RE.match(cleaned):
        return cleaned, []

    return raw_text, ["noncanonical_value"]


def _has_canonical_markers(raw_fact: Mapping[str, Any]) -> bool:
    return any(key in raw_fact for key in ("comment", "is_note", "note_reference"))


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
        comment_raw = payload.get("comment")
        if comment_raw in ("", None) and has_legacy:
            comment_raw = payload.get("note")
        note_raw = payload.get("note")
        if note_raw in ("", None) and has_legacy:
            note_raw = payload.get("beur_num")
    else:
        comment_raw = payload.get("note")
        note_raw = payload.get("beur_num")

    note_reference_raw = payload.get("note_reference")
    if note_reference_raw in ("", None):
        note_reference_raw = payload.get("refference", payload.get("reference", payload.get("ref")))
    note_reference = str(note_reference_raw or "").strip()

    is_note_raw = payload.get("is_note")
    if is_note_raw in ("", None):
        is_note_raw = payload.get("is_beur", payload.get("beur"))
    is_note, bool_warnings = _coerce_is_note(is_note_raw)

    value, value_warnings = normalize_value(payload.get("value"))
    date_value, date_warnings = normalize_date(payload.get("date"))

    normalized: dict[str, Any] = {
        "value": value,
        "comment": _to_optional_text(comment_raw),
        "is_note": is_note,
        "note": _to_optional_text(note_raw),
        "note_reference": note_reference,
        "date": date_value,
        "path": _normalize_path(payload.get("path")),
        "currency": _normalize_currency(payload.get("currency")),
        "scale": _normalize_scale(payload.get("scale")),
        "value_type": _normalize_value_type(payload.get("value_type")),
    }
    if include_bbox and "bbox" in payload:
        normalized["bbox"] = payload.get("bbox")

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


def _resolve_annotation_files(root: Path, annotations_glob: str) -> list[Path]:
    root = root.resolve()
    pattern = Path(annotations_glob).expanduser()
    search_pattern = str(pattern) if pattern.is_absolute() else str(root / annotations_glob)
    resolved = sorted(Path(path).resolve() for path in glob.glob(search_pattern, recursive=True))
    return [path for path in resolved if path.is_file()]


def fact_format_report(
    root: Path,
    *,
    annotations_glob: str = DEFAULT_ANNOTATIONS_GLOB,
) -> dict[str, Any]:
    root = root.resolve()
    files = _resolve_annotation_files(root, annotations_glob)
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
) -> dict[str, Any]:
    report = fact_format_report(root, annotations_glob=annotations_glob)
    summary = {
        "annotations_glob": report["annotations_glob"],
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
    "normalize_value",
]

