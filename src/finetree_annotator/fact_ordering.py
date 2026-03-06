from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal

from .schemas import (
    _normalize_company_id_value,
    _normalize_company_name_value,
    _normalize_doc_language_value,
    _normalize_reading_direction_value,
    _normalize_report_year_value,
)

Direction = Literal["rtl", "ltr"]
Language = Literal["he", "en"]
DirectionSource = Literal[
    "document_meta.reading_direction",
    "document_meta.language",
    "auto",
    "default",
]

DEFAULT_DIRECTION: Direction = "rtl"
DEFAULT_ROW_TOLERANCE_RATIO = 0.35
DEFAULT_ROW_TOLERANCE_MIN_PX = 6.0


def normalize_document_meta(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "language": None,
            "reading_direction": None,
            "company_name": None,
            "company_id": None,
            "report_year": None,
        }

    try:
        language = _normalize_doc_language_value(raw.get("language"))
    except ValueError:
        language = None
    try:
        direction = _normalize_reading_direction_value(raw.get("reading_direction"))
    except ValueError:
        direction = None
    try:
        company_name = _normalize_company_name_value(raw.get("company_name"))
    except ValueError:
        company_name = None
    try:
        company_id = _normalize_company_id_value(raw.get("company_id"))
    except ValueError:
        company_id = None
    try:
        report_year = _normalize_report_year_value(raw.get("report_year"))
    except ValueError:
        report_year = None
    return {
        "language": language,
        "reading_direction": direction,
        "company_name": company_name,
        "company_id": company_id,
        "report_year": report_year,
    }


def compact_document_meta(raw: Any) -> dict[str, Any]:
    normalized = normalize_document_meta(raw)
    out: dict[str, Any] = {}
    if normalized["language"]:
        out["language"] = str(normalized["language"])
    if normalized["reading_direction"]:
        out["reading_direction"] = str(normalized["reading_direction"])
    if normalized.get("company_name"):
        out["company_name"] = str(normalized["company_name"])
    if normalized.get("company_id"):
        out["company_id"] = str(normalized["company_id"])
    if normalized.get("report_year") is not None:
        out["report_year"] = int(normalized["report_year"])
    return out


def direction_from_language(language: str | None) -> Direction | None:
    if language == "he":
        return "rtl"
    if language == "en":
        return "ltr"
    return None


def _normalize_bbox(raw_bbox: Any) -> dict[str, float] | None:
    if isinstance(raw_bbox, dict):
        x_raw = raw_bbox.get("x", 0.0)
        y_raw = raw_bbox.get("y", 0.0)
        w_raw = raw_bbox.get("w", 1.0)
        h_raw = raw_bbox.get("h", 1.0)
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        x_raw, y_raw, w_raw, h_raw = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    else:
        return None
    try:
        x = float(x_raw)
        y = float(y_raw)
        w = max(float(w_raw), 1.0)
        h = max(float(h_raw), 1.0)
    except Exception:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _iter_pages_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    pages = payload.get("pages")
    if isinstance(pages, list):
        return [page for page in pages if isinstance(page, dict)]
    facts = payload.get("facts")
    if isinstance(facts, list):
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        image = str(payload.get("image") or "page")
        return [{"image": image, "meta": meta, "facts": facts}]
    return []


def _collect_text_parts(payload: Any) -> list[str]:
    texts: list[str] = []
    for page in _iter_pages_from_payload(payload):
        meta = page.get("meta") if isinstance(page.get("meta"), dict) else {}
        for key in ("entity_name", "page_num", "title", "type"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        facts = page.get("facts")
        if not isinstance(facts, list):
            continue
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            for key in ("value", "comment", "note_name", "note_reference", "date", "note_num", "note", "refference", "beur_num"):
                value = fact.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value)
            path = fact.get("path")
            if isinstance(path, list):
                for level in path:
                    level_text = str(level).strip()
                    if level_text:
                        texts.append(level_text)
    return texts


def _hebrew_latin_counts(payload: Any) -> tuple[int, int]:
    hebrew = 0
    latin = 0
    for text in _collect_text_parts(payload):
        for ch in text:
            code = ord(ch)
            if 0x0590 <= code <= 0x05FF:
                hebrew += 1
            elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                latin += 1
    return hebrew, latin


def resolve_reading_direction(
    document_meta: Any,
    *,
    payload: Any = None,
    default_direction: Direction = DEFAULT_DIRECTION,
) -> dict[str, Any]:
    normalized_meta = normalize_document_meta(document_meta)
    explicit_direction = normalized_meta.get("reading_direction")
    if explicit_direction in {"rtl", "ltr"}:
        return {
            "direction": explicit_direction,
            "source": "document_meta.reading_direction",
            "uncertain": False,
            "hebrew_chars": None,
            "latin_chars": None,
        }

    explicit_language = normalized_meta.get("language")
    language_direction = direction_from_language(explicit_language)
    if language_direction is not None:
        return {
            "direction": language_direction,
            "source": "document_meta.language",
            "uncertain": False,
            "hebrew_chars": None,
            "latin_chars": None,
        }

    hebrew_chars, latin_chars = _hebrew_latin_counts(payload)
    if hebrew_chars > 0 or latin_chars > 0:
        if hebrew_chars == 0:
            return {
                "direction": "ltr",
                "source": "auto",
                "uncertain": False,
                "hebrew_chars": hebrew_chars,
                "latin_chars": latin_chars,
            }
        if latin_chars == 0:
            return {
                "direction": "rtl",
                "source": "auto",
                "uncertain": False,
                "hebrew_chars": hebrew_chars,
                "latin_chars": latin_chars,
            }
        ratio = float(hebrew_chars) / max(float(latin_chars), 1.0)
        if ratio >= 1.2:
            return {
                "direction": "rtl",
                "source": "auto",
                "uncertain": False,
                "hebrew_chars": hebrew_chars,
                "latin_chars": latin_chars,
            }
        if ratio <= (1.0 / 1.2):
            return {
                "direction": "ltr",
                "source": "auto",
                "uncertain": False,
                "hebrew_chars": hebrew_chars,
                "latin_chars": latin_chars,
            }

    return {
        "direction": default_direction,
        "source": "default",
        "uncertain": True,
        "hebrew_chars": hebrew_chars,
        "latin_chars": latin_chars,
    }


def _row_tolerance(
    heights: Iterable[float],
    *,
    row_tolerance_ratio: float,
    row_tolerance_min_px: float,
) -> float:
    height_list = [float(h) for h in heights if float(h) > 0.0]
    if not height_list:
        return max(1.0, float(row_tolerance_min_px))
    return max(float(row_tolerance_min_px), float(median(height_list)) * float(row_tolerance_ratio))


def canonical_fact_order_indices(
    facts: list[dict[str, Any]],
    *,
    direction: Direction,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    row_tolerance_min_px: float = DEFAULT_ROW_TOLERANCE_MIN_PX,
) -> list[int]:
    if len(facts) <= 1:
        return list(range(len(facts)))

    entries: list[dict[str, float | int]] = []
    invalid_indexes: list[int] = []
    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            invalid_indexes.append(idx)
            continue
        bbox = _normalize_bbox(fact.get("bbox"))
        if bbox is None:
            invalid_indexes.append(idx)
            continue
        entries.append(
            {
                "idx": idx,
                "cx": bbox["x"] + (bbox["w"] / 2.0),
                "cy": bbox["y"] + (bbox["h"] / 2.0),
                "h": bbox["h"],
            }
        )

    if not entries:
        return list(range(len(facts)))

    tolerance = _row_tolerance(
        (float(e["h"]) for e in entries),
        row_tolerance_ratio=row_tolerance_ratio,
        row_tolerance_min_px=row_tolerance_min_px,
    )
    by_y = sorted(entries, key=lambda item: (float(item["cy"]), int(item["idx"])))

    rows: list[list[dict[str, float | int]]] = []
    current: list[dict[str, float | int]] = []
    current_anchor = 0.0
    for item in by_y:
        item_cy = float(item["cy"])
        if not current:
            current = [item]
            current_anchor = item_cy
            continue
        if abs(item_cy - current_anchor) <= tolerance:
            current.append(item)
            current_anchor = sum(float(v["cy"]) for v in current) / float(len(current))
            continue
        rows.append(current)
        current = [item]
        current_anchor = item_cy
    if current:
        rows.append(current)

    ordered_indexes: list[int] = []
    for row in rows:
        if direction == "ltr":
            row_sorted = sorted(row, key=lambda item: (float(item["cx"]), float(item["cy"]), int(item["idx"])))
        else:
            row_sorted = sorted(row, key=lambda item: (-float(item["cx"]), float(item["cy"]), int(item["idx"])))
        ordered_indexes.extend(int(item["idx"]) for item in row_sorted)

    seen = set(ordered_indexes)
    ordered_indexes.extend(idx for idx in invalid_indexes if idx not in seen)
    seen = set(ordered_indexes)
    ordered_indexes.extend(idx for idx in range(len(facts)) if idx not in seen)
    return ordered_indexes


def reorder_facts(
    facts: list[dict[str, Any]],
    *,
    direction: Direction,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    row_tolerance_min_px: float = DEFAULT_ROW_TOLERANCE_MIN_PX,
) -> list[dict[str, Any]]:
    expected = canonical_fact_order_indices(
        facts,
        direction=direction,
        row_tolerance_ratio=row_tolerance_ratio,
        row_tolerance_min_px=row_tolerance_min_px,
    )
    return [facts[idx] for idx in expected]


def validate_fact_order(
    facts: list[dict[str, Any]],
    *,
    direction: Direction,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    row_tolerance_min_px: float = DEFAULT_ROW_TOLERANCE_MIN_PX,
) -> dict[str, Any]:
    expected = canonical_fact_order_indices(
        facts,
        direction=direction,
        row_tolerance_ratio=row_tolerance_ratio,
        row_tolerance_min_px=row_tolerance_min_px,
    )
    violations: list[dict[str, Any]] = []
    for position, expected_idx in enumerate(expected):
        if expected_idx == position:
            continue
        current_fact = facts[position] if 0 <= position < len(facts) and isinstance(facts[position], dict) else {}
        expected_fact = facts[expected_idx] if 0 <= expected_idx < len(facts) and isinstance(facts[expected_idx], dict) else {}
        violations.append(
            {
                "position": position,
                "current_index": position,
                "expected_index": expected_idx,
                "current_bbox": _normalize_bbox(current_fact.get("bbox")),
                "expected_bbox": _normalize_bbox(expected_fact.get("bbox")),
                "current_value": str(current_fact.get("value") or ""),
                "expected_value": str(expected_fact.get("value") or ""),
            }
        )

    return {
        "ok": len(violations) == 0,
        "expected_indices": expected,
        "violations": violations,
    }


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


def fact_order_report(
    root: Path,
    *,
    annotations_glob: str = "data/annotations/*.json",
    default_direction: Direction = DEFAULT_DIRECTION,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    row_tolerance_min_px: float = DEFAULT_ROW_TOLERANCE_MIN_PX,
    include_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    uncertain_documents: list[dict[str, Any]] = []
    files_scanned = 0
    pages_scanned = 0
    pages_with_order_issues = 0

    for file_path in _resolve_annotation_files(root, annotations_glob, include_doc_ids=include_doc_ids):
        files_scanned += 1
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        doc_meta = payload.get("document_meta") if isinstance(payload, dict) else None
        direction_info = resolve_reading_direction(doc_meta, payload=payload, default_direction=default_direction)
        direction = str(direction_info["direction"])
        pages = _iter_pages_from_payload(payload)
        if bool(direction_info.get("uncertain")):
            uncertain_documents.append(
                {
                    "file": str(file_path.relative_to(root.resolve())),
                    "direction": direction,
                    "source": str(direction_info.get("source") or ""),
                    "hebrew_chars": direction_info.get("hebrew_chars"),
                    "latin_chars": direction_info.get("latin_chars"),
                }
            )
        for page_idx, page in enumerate(pages):
            pages_scanned += 1
            facts = page.get("facts")
            if not isinstance(facts, list):
                continue
            typed_facts = [fact for fact in facts if isinstance(fact, dict)]
            validation = validate_fact_order(
                typed_facts,
                direction="rtl" if direction == "rtl" else "ltr",
                row_tolerance_ratio=row_tolerance_ratio,
                row_tolerance_min_px=row_tolerance_min_px,
            )
            if bool(validation["ok"]):
                continue
            pages_with_order_issues += 1
            findings.append(
                {
                    "file": str(file_path.relative_to(root.resolve())),
                    "page": str(page.get("image") or f"page_{page_idx + 1}"),
                    "direction": direction,
                    "direction_source": str(direction_info.get("source") or ""),
                    "violations_count": len(validation["violations"]),
                    "violations": validation["violations"],
                    "first_violation": validation["violations"][0] if validation["violations"] else None,
                }
            )

    return {
        "annotations_glob": annotations_glob,
        "include_doc_ids": sorted(_normalize_included_doc_ids(include_doc_ids) or set()),
        "files_scanned": files_scanned,
        "pages_scanned": pages_scanned,
        "pages_with_order_issues": pages_with_order_issues,
        "uncertain_documents": uncertain_documents,
        "findings": findings,
    }


def assert_fact_order(
    root: Path,
    *,
    annotations_glob: str = "data/annotations/*.json",
    default_direction: Direction = DEFAULT_DIRECTION,
    row_tolerance_ratio: float = DEFAULT_ROW_TOLERANCE_RATIO,
    row_tolerance_min_px: float = DEFAULT_ROW_TOLERANCE_MIN_PX,
    fail_on_issues: bool = True,
    include_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    report = fact_order_report(
        root,
        annotations_glob=annotations_glob,
        default_direction=default_direction,
        row_tolerance_ratio=row_tolerance_ratio,
        row_tolerance_min_px=row_tolerance_min_px,
        include_doc_ids=include_doc_ids,
    )
    findings_preview = [
        {
            "file": finding.get("file"),
            "page": finding.get("page"),
            "direction": finding.get("direction"),
            "direction_source": finding.get("direction_source"),
            "violations_count": finding.get("violations_count"),
            "first_violation": finding.get("first_violation"),
        }
        for finding in report["findings"][:5]
    ]
    print(
        "FACT_ORDER_AUDIT:",
        json.dumps(
            {
                "annotations_glob": report["annotations_glob"],
                "include_doc_ids": report["include_doc_ids"],
                "files_scanned": report["files_scanned"],
                "pages_scanned": report["pages_scanned"],
                "pages_with_order_issues": report["pages_with_order_issues"],
                "uncertain_documents": len(report["uncertain_documents"]),
                "findings_preview": findings_preview,
            },
            ensure_ascii=False,
        ),
    )
    if fail_on_issues and int(report["pages_with_order_issues"]) > 0:
        raise RuntimeError(
            "Fact reading-order violations detected in annotation JSON. "
            f"pages_with_order_issues={report['pages_with_order_issues']}. "
            "Run scripts/check_fact_reading_order.py for details, or pass --allow-ordering-issues to bypass."
        )
    return report


__all__ = [
    "DEFAULT_DIRECTION",
    "DEFAULT_ROW_TOLERANCE_MIN_PX",
    "DEFAULT_ROW_TOLERANCE_RATIO",
    "assert_fact_order",
    "canonical_fact_order_indices",
    "compact_document_meta",
    "direction_from_language",
    "fact_order_report",
    "normalize_document_meta",
    "reorder_facts",
    "resolve_reading_direction",
    "validate_fact_order",
]
