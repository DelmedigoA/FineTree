from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Sequence

from .annotation_core import make_placeholder_bbox, normalize_bbox_data, normalize_fact_data
from .fact_ordering import canonical_fact_geometry_rows
from .numeric_text import normalize_angle_bracketed_numeric_text

_CURRENCY_SIGILS = {"$", "EUR", "£", "₪", "€"}
_DASH_CHARS_RE = re.compile(r"[\u2212\u2013\u2014\u2012\u2011\u2010]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z%+\-().]")


def normalize_bbox_match_value(raw_value: Any) -> str:
    text = normalize_angle_bracketed_numeric_text(raw_value)
    text = str(text or "").strip()
    if not text:
        return ""

    text = _DASH_CHARS_RE.sub("-", text)
    text = text.replace("\u2009", " ").replace("\xa0", " ")
    compact = re.sub(r"\s+", "", text)
    for sigil in _CURRENCY_SIGILS:
        compact = compact.replace(sigil, "")
    compact = compact.strip(".,:;")
    if compact and set(compact) == {"-"}:
        return "num:-"

    suffix = ""
    negative = False
    if compact.endswith("%"):
        suffix = "%"
        compact = compact[:-1]
    if compact.startswith("(") and compact.endswith(")"):
        negative = True
        compact = compact[1:-1]
    if compact.startswith("+"):
        compact = compact[1:]
    elif compact.startswith("-"):
        negative = True
        compact = compact[1:]
    compact = compact.strip(".,:;").replace(",", "")
    if compact and re.fullmatch(r"\d+(?:\.\d+)?", compact):
        try:
            number = Decimal(compact)
            if negative:
                number = -number
            return f"num:{_format_decimal_plain(number)}{suffix}"
        except InvalidOperation:
            pass
    prefix = "-" if negative else ""
    lowered = _NON_ALNUM_RE.sub("", compact).lower()
    return f"text:{prefix}{lowered}{suffix}"


@dataclass(frozen=True)
class FactCandidate:
    global_index: int
    payload: dict[str, Any]
    value: str
    match_key: str


def match_qwen_import_payloads(
    *,
    page_name: str,
    imported_payloads: list[dict[str, Any]],
    detector_payloads: list[dict[str, Any]],
    reading_direction: str,
    placeholder_bbox_factory: Callable[[int], dict[str, float]] = make_placeholder_bbox,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_imported_payloads = [deepcopy(payload) for payload in imported_payloads]
    ordered_imported_payloads = _ordered_imported_payloads(imported_payloads)
    ordered_detector_payloads = _ordered_detector_payloads(
        detector_payloads,
        reading_direction=reading_direction,
    )
    imported_facts = _build_fact_candidates(ordered_imported_payloads)
    detector_facts = _build_fact_candidates(ordered_detector_payloads)

    matched_detector_indices_by_imported_index: dict[int, int] = {}
    detector_index_by_imported_index: dict[int, int] = {}
    row_matches: list[dict[str, Any]] = []
    available_detector_facts = [deepcopy(fact) for fact in detector_facts]

    for fact in imported_facts:
        key = fact.match_key
        if not key:
            row_matches.append(_unmatched_debug_row(fact, reason="missing_value"))
            continue
        detector_pool_index = next(
            (idx for idx, detector_fact in enumerate(available_detector_facts) if detector_fact.match_key == key),
            None,
        )
        if detector_pool_index is None:
            row_matches.append(_unmatched_debug_row(fact, reason="detector_missing"))
            continue
        detector_fact = available_detector_facts.pop(detector_pool_index)
        matched_detector_indices_by_imported_index[fact.global_index] = detector_fact.global_index
        detector_index_by_imported_index[fact.global_index] = detector_fact.global_index
        row_matches.append(
            {
                "matched": True,
                "reason": "ordered_pool_value_match",
                "import_indices": [fact.global_index],
                "import_values": [fact.value],
                "detector_indices": [detector_fact.global_index],
                "detector_values": [detector_fact.value],
                "matched_count": 1,
                "match_key": key,
                "ordered_detector_pool_index": int(detector_pool_index),
            }
        )

    merged_payloads = [deepcopy(payload) for payload in ordered_imported_payloads]
    unmatched_import_indices = [
        index
        for index in range(len(ordered_imported_payloads))
        if index not in matched_detector_indices_by_imported_index
    ]
    for placeholder_index, import_index in enumerate(unmatched_import_indices):
        merged_payloads[int(import_index)]["bbox"] = normalize_bbox_data(placeholder_bbox_factory(placeholder_index))
    for import_index, detector_index in matched_detector_indices_by_imported_index.items():
        merged_payloads[int(import_index)]["bbox"] = normalize_bbox_data(
            ordered_detector_payloads[int(detector_index)].get("bbox")
        )

    matched_detector_indices = set(matched_detector_indices_by_imported_index.values())
    unmatched_detector_indices = [
        index for index in range(len(ordered_detector_payloads)) if index not in matched_detector_indices
    ]
    matches = [
        {
            "gemini_index": int(import_index),
            "detector_index": int(detector_index),
            "gemini_value": normalize_fact_data(ordered_imported_payloads[int(import_index)]).get("value"),
            "detector_value": normalize_fact_data(ordered_detector_payloads[int(detector_index)]).get("value"),
            "match_key": normalize_bbox_match_value(
                normalize_fact_data(ordered_imported_payloads[int(import_index)]).get("value") or ""
            ),
        }
        for import_index, detector_index in sorted(matched_detector_indices_by_imported_index.items())
    ]
    debug_payload = {
        "page_name": page_name,
        "gemini_count": len(ordered_imported_payloads),
        "detector_count": len(ordered_detector_payloads),
        "matched_count": len(matches),
        "ordered_gemini_payloads": [deepcopy(payload) for payload in ordered_imported_payloads],
        "ordered_detector_payloads": [deepcopy(payload) for payload in ordered_detector_payloads],
        "raw_imported_payloads": raw_imported_payloads,
        "matches": matches,
        "unmatched_gemini_indices": unmatched_import_indices,
        "unmatched_detector_indices": unmatched_detector_indices,
        "ambiguous_gemini_indices": [],
        "ambiguous_detector_indices": [],
        "ignored_detector_indices": [],
        "imported_row_groups": _summarize_facts(imported_facts),
        "detector_row_groups": _summarize_facts(detector_facts),
        "available_detector_pool_after_match": _summarize_facts(available_detector_facts),
        "row_matches": row_matches,
        "column_clusters": [],
        "row_match_strategy": "ordered_pool_value_match",
        "reading_direction": reading_direction,
        "placeholder_count": len(unmatched_import_indices),
        "detector_index_by_gemini_index": {
            str(import_index): int(detector_index)
            for import_index, detector_index in sorted(detector_index_by_imported_index.items())
        },
    }
    return merged_payloads, debug_payload


def _ordered_imported_payloads(imported_payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = [deepcopy(payload) for payload in imported_payloads]
    fact_nums = [payload.get("fact_num") for payload in ordered]
    if fact_nums and all(isinstance(fact_num, int) and fact_num >= 1 for fact_num in fact_nums):
        ordered.sort(key=lambda payload: int(payload.get("fact_num") or 0))
    return ordered


def _ordered_detector_payloads(
    detector_payloads: Sequence[dict[str, Any]],
    *,
    reading_direction: str,
) -> list[dict[str, Any]]:
    payloads = [deepcopy(payload) for payload in detector_payloads]
    rows = canonical_fact_geometry_rows(
        [{"bbox": normalize_bbox_data(payload.get("bbox"))} for payload in payloads],
        direction="rtl" if reading_direction == "rtl" else "ltr",
        row_tolerance_ratio=0.35,
        row_tolerance_min_px=6.0,
    )
    if not rows:
        return payloads
    ordered: list[dict[str, Any]] = []
    for row in rows:
        for index in row["indices"]:
            if 0 <= int(index) < len(payloads):
                ordered.append(deepcopy(payloads[int(index)]))
    return ordered


def _build_fact_candidates(payloads: list[dict[str, Any]]) -> list[FactCandidate]:
    return [
        FactCandidate(
            global_index=int(index),
            payload=deepcopy(payload),
            value=str(normalize_fact_data(payload).get("value") or ""),
            match_key=normalize_bbox_match_value(normalize_fact_data(payload).get("value") or ""),
        )
        for index, payload in enumerate(payloads)
    ]


def _summarize_facts(facts: list[FactCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": int(fact.global_index),
            "indices": [int(fact.global_index)],
            "values": [fact.value],
            "match_keys": [fact.match_key],
        }
        for fact in facts
    ]


def _unmatched_debug_row(fact: FactCandidate, *, reason: str) -> dict[str, Any]:
    return {
        "matched": False,
        "reason": reason,
        "import_indices": [fact.global_index],
        "import_values": [fact.value],
        "detector_indices": [],
        "detector_values": [],
        "matched_count": 0,
        "match_key": fact.match_key,
    }


def _format_decimal_plain(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


__all__ = [
    "match_qwen_import_payloads",
    "normalize_bbox_match_value",
]
