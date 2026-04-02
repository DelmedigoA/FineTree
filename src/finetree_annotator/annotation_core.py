"""Core data helpers for the PDF annotator.

This module intentionally has no Qt dependency so it can be unit tested.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from .bbox_utils import bbox_to_list, denormalize_bbox_from_1000, normalize_bbox_data
from .equation_integrity import audit_and_rebuild_financial_facts
from .equation_integrity import resequence_fact_numbers_and_remap_fact_equations
from .fact_ordering import compact_document_meta, normalize_document_meta
from .fact_normalization import normalize_fact_payload
from .schema_io import canonicalize_with_findings, save_canonical
from .schema_contract import CANONICAL_FACT_KEYS, CURRENCY_VALUES, SCALE_VALUES
from .schemas import Fact, PageMeta, PageType
from .vision_resize import restore_bbox_from_resized_pixels_with_stats

FACT_KEYS = tuple(CANONICAL_FACT_KEYS)
CURRENCY_OPTIONS = list(CURRENCY_VALUES)
SCALE_OPTIONS = list(SCALE_VALUES)
IMPORT_BBOX_MODE_ORIGINAL_PIXELS = "original_pixels"
IMPORT_BBOX_MODE_NORMALIZED_1000 = "normalized_1000"
IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS = "resized_pixels_via_max_pixels"
IMPORT_PLACEHOLDER_BBOX_WIDTH = 72.0
IMPORT_PLACEHOLDER_BBOX_HEIGHT = 28.0
_IMPORT_PLACEHOLDER_BBOX_PADDING = 8.0
_IMPORT_PLACEHOLDER_BBOX_X_STEP = 80.0
_IMPORT_PLACEHOLDER_BBOX_Y_STEP = 36.0
_IMPORT_PLACEHOLDER_BBOX_COLUMNS = 2
_IMPORT_PLACEHOLDER_BBOX_FLAG = "__ft_import_placeholder_bbox__"


@dataclass
class BoxRecord:
    bbox: Dict[str, float]
    fact: Dict[str, Any]


@dataclass
class PageState:
    meta: Dict[str, Any] = field(default_factory=dict)
    facts: List[BoxRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ImportBBoxConversionStats:
    converted: int = 0
    clamped: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class ImportJsonParseResult:
    payload: Any
    recovered: bool = False
    message: str | None = None


@dataclass(frozen=True)
class PageTextCorrectionValidationResult:
    accepted: bool
    changed_paths: tuple[str, ...] = ()
    changed_fields: tuple[dict[str, Any], ...] = ()
    rejection_reason: str | None = None


def default_fact_data() -> Dict[str, Any]:
    return {
        "value": "",
        "fact_num": None,
        "equations": None,
        "natural_sign": None,
        "row_role": "detail",
        "comment_ref": None,
        "note_flag": False,
        "note_name": None,
        "note_num": None,
        "note_ref": None,
        "date": None,
        "period_type": None,
        "period_start": None,
        "period_end": None,
        "duration_type": None,
        "recurring_period": None,
        "path": [],
        "path_source": None,
        "currency": None,
        "scale": None,
        "value_type": None,
        "value_context": None,
    }


def _assign_missing_fact_numbers(facts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_facts = [normalize_fact_data(raw_fact) for raw_fact in facts]
    return resequence_fact_numbers_and_remap_fact_equations(normalized_facts)


def normalize_fact_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized, _warnings = normalize_fact_payload(data or {}, include_bbox=False)
    merged = {**default_fact_data(), **normalized}
    return merged


def default_page_meta(index: int = 0) -> Dict[str, Any]:
    return {
        "entity_name": None,
        "page_num": None,
        "page_type": PageType.other.value,
        "statement_type": None,
        "title": None,
        "annotation_note": None,
        "annotation_status": None,
    }


def _coerce_raw_fact(raw_fact: Dict[str, Any]) -> Dict[str, Any]:
    if "fact" in raw_fact and isinstance(raw_fact["fact"], dict):
        return normalize_fact_data(raw_fact["fact"])
    return normalize_fact_data(raw_fact)


def make_placeholder_bbox(index: int = 0) -> Dict[str, float]:
    safe_index = max(int(index), 0)
    col = safe_index % _IMPORT_PLACEHOLDER_BBOX_COLUMNS
    row = safe_index // _IMPORT_PLACEHOLDER_BBOX_COLUMNS
    return normalize_bbox_data(
        {
            "x": _IMPORT_PLACEHOLDER_BBOX_PADDING + (col * _IMPORT_PLACEHOLDER_BBOX_X_STEP),
            "y": _IMPORT_PLACEHOLDER_BBOX_PADDING + (row * _IMPORT_PLACEHOLDER_BBOX_Y_STEP),
            "w": IMPORT_PLACEHOLDER_BBOX_WIDTH,
            "h": IMPORT_PLACEHOLDER_BBOX_HEIGHT,
        }
    )


def _normalize_import_page_meta(raw_meta: Any, page_index: int) -> Dict[str, Any]:
    defaults = default_page_meta(page_index)
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    try:
        return PageMeta.model_validate({**defaults, **meta}).model_dump(mode="json")
    except ValidationError:
        normalized_meta = dict(defaults)
        for key, value in meta.items():
            try:
                normalized_meta = PageMeta.model_validate(
                    {**defaults, **normalized_meta, key: value}
                ).model_dump(mode="json")
            except ValidationError:
                continue
        return PageMeta.model_validate({**defaults, **normalized_meta}).model_dump(mode="json")


def _normalize_import_bbox_or_placeholder(
    raw_bbox: Any,
    *,
    placeholder_index: int,
) -> tuple[Dict[str, float], bool]:
    if isinstance(raw_bbox, dict):
        bbox_candidate: Any = raw_bbox
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        bbox_candidate = raw_bbox
    else:
        return make_placeholder_bbox(placeholder_index), True
    try:
        return normalize_bbox_data(bbox_candidate), False
    except Exception:
        return make_placeholder_bbox(placeholder_index), True


def load_page_states(
    payload: Dict[str, Any],
    page_image_names: Iterable[str],
    *,
    placeholder_missing_bboxes: bool = False,
) -> Dict[str, PageState]:
    page_names = list(page_image_names)
    available = set(page_names)
    index_map = {name: idx for idx, name in enumerate(page_names)}
    states: Dict[str, PageState] = {}
    pages = payload.get("pages", [])
    if not isinstance(pages, list):
        return states

    for page in pages:
        if not isinstance(page, dict):
            continue
        page_name = page.get("image")
        if page_name not in available:
            continue

        raw_meta = page.get("meta")
        page_index = index_map.get(str(page_name), 0)
        meta_model = _normalize_import_page_meta(raw_meta, page_index)
        raw_facts = page.get("facts", [])
        facts: List[BoxRecord] = []
        if isinstance(raw_facts, list):
            raw_fact_dicts = [fact for fact in raw_facts if isinstance(fact, dict)]
            normalized_facts = _assign_missing_fact_numbers([_coerce_raw_fact(fact) for fact in raw_fact_dicts])
            for fact_index, (raw_fact, normalized_fact) in enumerate(zip(raw_fact_dicts, normalized_facts)):
                if placeholder_missing_bboxes:
                    bbox, used_placeholder = _normalize_import_bbox_or_placeholder(
                        raw_fact.get("bbox"),
                        placeholder_index=fact_index,
                    )
                    fact_payload = dict(normalized_fact)
                    if used_placeholder:
                        fact_payload[_IMPORT_PLACEHOLDER_BBOX_FLAG] = True
                else:
                    bbox = normalize_bbox_data(raw_fact.get("bbox"))
                    fact_payload = normalized_fact
                facts.append(BoxRecord(bbox=bbox, fact=fact_payload))
        states[str(page_name)] = PageState(meta=meta_model, facts=facts)
    return states


def parse_import_payload(
    payload: Any,
    page_image_names: Sequence[str],
    default_page_image_name: Optional[str],
) -> Dict[str, PageState]:
    document_payload = normalize_import_payload_to_document(
        payload,
        page_image_names=page_image_names,
        default_page_image_name=default_page_image_name,
    )
    return load_page_states(document_payload, page_image_names, placeholder_missing_bboxes=True)


def _clean_json_candidate(candidate: str) -> str:
    fixed = candidate.strip()
    if "\n" in fixed:
        first_line, rest = fixed.split("\n", 1)
        if first_line.strip().lower() in {"json", "javascript", "js"}:
            fixed = rest.strip()
    fixed = fixed.replace("“", "\"").replace("”", "\"").replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _extract_balanced_block_from_index(text: str, start_idx: int, open_char: str, close_char: str) -> str | None:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != open_char:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _parse_json_like(candidate: str) -> Any:
    variants: list[str] = []
    for variant in (candidate, _clean_json_candidate(candidate)):
        if variant not in variants:
            variants.append(variant)
        trimmed = variant.rstrip().rstrip(",").strip()
        if trimmed and trimmed not in variants:
            variants.append(trimmed)
    for variant in variants:
        try:
            parsed = json.loads(variant)
            if isinstance(parsed, str):
                nested = str(parsed).strip()
                if nested and nested != variant:
                    try:
                        return _parse_json_like(nested)
                    except Exception:
                        pass
            return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(variant)
            if isinstance(parsed, str):
                nested = str(parsed).strip()
                if nested and nested != variant:
                    try:
                        return _parse_json_like(nested)
                    except Exception:
                        pass
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass
    raise ValueError("Could not parse JSON-like payload.")


def _extract_key_array_start(text: str, key: str) -> int:
    for token in (f'"{key}"', f"'{key}'"):
        idx = text.find(token)
        if idx < 0:
            continue
        colon = text.find(":", idx + len(token))
        if colon < 0:
            continue
        array_start = text.find("[", colon + 1)
        if array_start >= 0:
            return array_start
    return -1


def _extract_key_object(text: str, key: str) -> dict[str, Any] | None:
    for token in (f'"{key}"', f"'{key}'"):
        idx = text.find(token)
        if idx < 0:
            continue
        colon = text.find(":", idx + len(token))
        if colon < 0:
            continue
        brace = text.find("{", colon + 1)
        if brace < 0:
            continue
        block = _extract_balanced_block_from_index(text, brace, "{", "}")
        if not block:
            continue
        try:
            parsed = _parse_json_like(block)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_key_string(text: str, key: str) -> str | None:
    for token in (f'"{key}"', f"'{key}'"):
        idx = text.find(token)
        if idx < 0:
            continue
        colon = text.find(":", idx + len(token))
        if colon < 0:
            continue
        match = re.search(r'"((?:\\.|[^"\\])*)"', text[colon + 1 :])
        if match is None:
            continue
        try:
            return json.loads(f'"{match.group(1)}"')
        except Exception:
            continue
    return None


def _extract_key_object_by_anchor(text: str, key: str, next_keys: Sequence[str]) -> dict[str, Any] | None:
    for token in (f'"{key}"', f"'{key}'"):
        idx = text.find(token)
        if idx < 0:
            continue
        colon = text.find(":", idx + len(token))
        if colon < 0:
            continue
        brace = text.find("{", colon + 1)
        if brace < 0:
            continue
        end = len(text)
        for next_key in next_keys:
            for next_token in (f'"{next_key}"', f"'{next_key}'"):
                next_idx = text.find(next_token, brace + 1)
                if next_idx > 0:
                    end = min(end, next_idx)
        candidate = text[brace:end].rstrip().rstrip(",")
        if candidate and not candidate.endswith("}"):
            candidate = candidate + "}"
        try:
            parsed = _parse_json_like(candidate)
        except Exception:
            repaired_candidate = candidate.replace(r'\\\\\"', r'\\\"')
            if repaired_candidate != candidate:
                try:
                    parsed = _parse_json_like(repaired_candidate)
                except Exception:
                    continue
            else:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_page_meta_from_text(text: str) -> dict[str, Any] | None:
    meta_match = re.search(
        r'"meta"\s*:\s*(\{.*?\})\s*,\s*"(?:facts|image|metadata|document_meta)"\s*:',
        text,
        re.S,
    )
    if not meta_match:
        return None

    meta_candidate = meta_match.group(1)
    for variant in (
        meta_candidate,
        meta_candidate.replace(r'\\\\\"', r'\\\"'),
    ):
        try:
            parsed_meta = _parse_json_like(variant)
        except Exception:
            continue
        if isinstance(parsed_meta, dict):
            return parsed_meta
    return None


def _extract_object_blocks_from_array(
    text: str,
    array_start: int,
) -> tuple[list[str], bool, int]:
    if array_start < 0 or array_start >= len(text) or text[array_start] != "[":
        return [], False, array_start
    blocks: list[str] = []
    pos = array_start + 1
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text):
            return blocks, False, pos
        if text[pos] == "]":
            return blocks, True, pos + 1
        if text[pos] != "{":
            next_obj = text.find("{", pos)
            next_close = text.find("]", pos)
            if next_obj < 0:
                return blocks, False, pos
            if next_close >= 0 and next_close < next_obj:
                return blocks, True, next_close + 1
            pos = next_obj
        block = _extract_balanced_block_from_index(text, pos, "{", "}")
        if not block:
            return blocks, False, pos
        blocks.append(block)
        pos += len(block)
    return blocks, False, pos


def _salvage_facts_from_text(text: str) -> list[dict[str, Any]]:
    facts_start = _extract_key_array_start(text, "facts")
    if facts_start < 0:
        return []
    fact_blocks, _array_closed, _end_pos = _extract_object_blocks_from_array(text, facts_start)
    facts: list[dict[str, Any]] = []
    for block in fact_blocks:
        try:
            parsed = _parse_json_like(block)
        except Exception:
            continue
        if isinstance(parsed, dict):
            facts.append(parsed)
    return facts


def _salvage_single_page_payload_from_text(text: str) -> dict[str, Any] | None:
    image_name = _extract_key_string(text, "image")
    meta = _extract_key_object(text, "meta")
    if meta is None:
        meta_match = re.search(
            r'"meta"\s*:\s*(\{.*?\})\s*,\s*"(?:facts|image|metadata|document_meta)"\s*:',
            text,
            re.S,
        )
        if meta_match:
            meta_candidate = meta_match.group(1)
            for variant in (
                meta_candidate,
                meta_candidate.replace(r'\\\\\"', r'\\\"'),
            ):
                try:
                    parsed_meta = _parse_json_like(variant)
                except Exception:
                    continue
                if isinstance(parsed_meta, dict):
                    meta = parsed_meta
                    break
    if meta is None:
        meta = _extract_key_object_by_anchor(text, "meta", ("facts", "image", "metadata", "document_meta"))
    meta = meta or {}
    facts = _salvage_facts_from_text(text)
    if image_name is None and not meta and not facts:
        return None
    payload: dict[str, Any] = {"meta": meta, "facts": facts}
    if image_name is not None:
        payload["image"] = image_name
    return payload


def _salvage_pages_payload_from_text(text: str) -> dict[str, Any] | None:
    pages_start = _extract_key_array_start(text, "pages")
    if pages_start < 0:
        return None
    page_blocks, _array_closed, end_pos = _extract_object_blocks_from_array(text, pages_start)
    pages: list[dict[str, Any]] = []
    for block in page_blocks:
        try:
            parsed = _parse_json_like(block)
        except Exception:
            continue
        if isinstance(parsed, dict):
            pages.append(parsed)
    partial_page = _salvage_single_page_payload_from_text(text[end_pos:])
    if partial_page is not None:
        pages.append(partial_page)
    if not pages:
        return None
    payload: dict[str, Any] = {"pages": pages}
    metadata = _extract_key_object(text, "metadata")
    if metadata:
        payload["metadata"] = metadata
    document_meta = _extract_key_object(text, "document_meta")
    if document_meta and "metadata" not in payload:
        payload["document_meta"] = document_meta
    images_dir = _extract_key_string(text, "images_dir")
    if images_dir is not None:
        payload["images_dir"] = images_dir
    return payload


def _salvage_page_sequence_payload_from_text(text: str) -> dict[str, Any] | None:
    line_items = _split_line_wrapped_list_items(text)
    if line_items:
        pages: list[dict[str, Any]] = []
        for index, item_text in enumerate(line_items, start=1):
            try:
                pages.extend(_coerce_page_payload_items(item_text, index=index))
            except ValueError:
                continue
        if pages:
            return {"pages": pages}

    top_level_items = _split_top_level_list_items(text)
    if top_level_items:
        pages: list[dict[str, Any]] = []
        for index, item_text in enumerate(top_level_items, start=1):
            try:
                pages.extend(_coerce_page_payload_items(item_text, index=index))
            except ValueError:
                continue
        if pages:
            return {"pages": pages}

    pages: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        brace = text.find("{", pos)
        if brace < 0:
            break
        block = _extract_balanced_block_from_index(text, brace, "{", "}")
        if not block:
            fragment_end_candidates = [
                idx for idx in (
                    text.find("\n", brace + 1),
                    text.find("',", brace + 1),
                    text.find("']", brace + 1),
                )
                if idx >= 0
            ]
            fragment_end = min(fragment_end_candidates) if fragment_end_candidates else len(text)
            salvaged_fragment = _salvage_single_page_payload_from_text(text[brace:fragment_end])
            if salvaged_fragment is not None:
                pages.append(salvaged_fragment)
                pos = fragment_end + 1
                continue
            pos = brace + 1
            continue
        try:
            parsed = _parse_json_like(block)
        except Exception:
            pos = brace + 1
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("pages"), list):
            pages.extend(page for page in parsed.get("pages", []) if isinstance(page, dict))
            pos = brace + len(block)
            continue
        if _looks_like_page_payload(parsed):
            pages.append(dict(parsed))
            pos = brace + len(block)
            continue
        pos = brace + 1
    if not pages:
        return None
    return {"pages": pages}


def _looks_like_page_payload(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if isinstance(item.get("pages"), list):
        return True
    return any(key in item for key in ("image", "meta", "facts"))


def _coerce_page_payload_items(item: Any, *, index: int) -> list[dict[str, Any]]:
    if isinstance(item, str):
        text = str(item).strip()
        if not text:
            return []
        try:
            parsed_item = _parse_json_like(text)
        except Exception:
            salvaged_page = _salvage_single_page_payload_from_text(text)
            if salvaged_page is not None:
                if not salvaged_page.get("meta"):
                    page_meta = _extract_page_meta_from_text(text)
                    if page_meta:
                        salvaged_page = {**salvaged_page, "meta": page_meta}
                return [salvaged_page]
            salvaged_pages = _salvage_page_sequence_payload_from_text(text)
            if salvaged_pages is not None and isinstance(salvaged_pages.get("pages"), list):
                pages = [page for page in salvaged_pages.get("pages", []) if isinstance(page, dict)]
                if pages:
                    if len(pages) == 1 and not pages[0].get("meta"):
                        page_meta = _extract_page_meta_from_text(text)
                        if page_meta:
                            pages = [{**pages[0], "meta": page_meta}]
                    return pages
            raise ValueError(f"Could not parse page {index} JSON string.")
        return _coerce_page_payload_items(parsed_item, index=index)

    if isinstance(item, list):
        normalized = _normalize_page_sequence_payload(item)
        if normalized is not None:
            return [page for page in normalized.get("pages", []) if isinstance(page, dict)]
        raise ValueError(f"List item {index} is not a page JSON object.")

    if isinstance(item, dict):
        if isinstance(item.get("pages"), list):
            nested_pages: list[dict[str, Any]] = []
            for nested_index, nested_item in enumerate(item.get("pages", []), start=1):
                nested_pages.extend(_coerce_page_payload_items(nested_item, index=nested_index))
            return nested_pages
        if _looks_like_page_payload(item):
            return [dict(item)]

    raise ValueError(f"List item {index} is not a page JSON object.")


def _split_top_level_list_items(text: str) -> list[str] | None:
    list_start = text.find("[")
    if list_start < 0:
        return None
    block = _extract_balanced_block_from_index(text, list_start, "[", "]")
    if block:
        inner = block[1:-1]
    else:
        inner = text[list_start + 1 :]
    items: list[str] = []
    start = 0
    depth = 0
    in_str = False
    quote_char = ""
    esc = False
    for idx, ch in enumerate(inner):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote_char:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote_char = ch
            continue
        if ch in "[{(":
            depth += 1
            continue
        if ch in "]})":
            if depth > 0:
                depth -= 1
            continue
        if ch == "," and depth == 0:
            item = inner[start:idx].strip()
            if item:
                items.append(item)
            start = idx + 1
    tail = inner[start:].strip()
    if tail:
        items.append(tail)
    return items


def _split_line_wrapped_list_items(text: str) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    items: list[str] = []
    for index, line in enumerate(lines):
        candidate = line
        if index == 0 and candidate.startswith("["):
            candidate = candidate[1:].lstrip()
        if index == len(lines) - 1 and candidate.endswith("]"):
            candidate = candidate[:-1].rstrip()
        candidate = candidate.rstrip(",").strip()
        if not candidate:
            continue
        if candidate[0] not in {"'", '"', "{", "["}:
            return None
        items.append(candidate)

    if len(items) < 2:
        return None
    return items


def _normalize_page_sequence_payload(payload: list[Any]) -> dict[str, Any] | None:
    is_page_sequence = any(isinstance(item, str) for item in payload) or any(
        _looks_like_page_payload(item) for item in payload
    )
    if not is_page_sequence:
        return None

    pages: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        pages.extend(_coerce_page_payload_items(item, index=index))

    if not pages:
        raise ValueError("No importable pages were found in the pasted list.")
    return {"pages": pages}


def _normalize_top_level_import_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        text = str(payload).strip()
        if text:
            try:
                return _normalize_top_level_import_payload(_parse_json_like(text))
            except Exception:
                return payload
    if isinstance(payload, list):
        normalized = _normalize_page_sequence_payload(payload)
        if normalized is not None:
            return normalized
    return payload


def _assign_page_images_by_order(
    payload: dict[str, Any],
    *,
    page_image_names: Sequence[str],
    default_page_image_name: Optional[str],
) -> dict[str, Any]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return payload

    available = set(page_image_names)
    single_page_target = (
        str(default_page_image_name).strip()
        if isinstance(default_page_image_name, str)
        and str(default_page_image_name).strip()
        and str(default_page_image_name).strip() in available
        and len(pages) == 1
        else None
    )
    assigned_pages: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        assigned = dict(page)
        image_name = assigned.get("image")
        if not isinstance(image_name, str) or not image_name.strip() or image_name not in available:
            if single_page_target is not None:
                assigned["image"] = single_page_target
            elif index < len(page_image_names):
                assigned["image"] = page_image_names[index]
        assigned_pages.append(assigned)
    return {**payload, "pages": assigned_pages}


def normalize_import_payload_to_document(
    payload: Any,
    page_image_names: Sequence[str],
    default_page_image_name: Optional[str],
) -> dict[str, Any]:
    normalized_payload = _normalize_top_level_import_payload(payload)
    if isinstance(normalized_payload, dict) and isinstance(normalized_payload.get("pages"), list):
        return _assign_page_images_by_order(
            normalized_payload,
            page_image_names=page_image_names,
            default_page_image_name=default_page_image_name,
        )

    if isinstance(normalized_payload, list):
        normalized_payload = {"meta": {}, "facts": normalized_payload}
    if not isinstance(normalized_payload, dict):
        return {"pages": []}

    default_image = default_page_image_name or (page_image_names[0] if page_image_names else None)
    if not default_image:
        return {"pages": []}

    image_name = normalized_payload.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        image_name = default_image

    page_obj = {
        "image": image_name,
        "meta": normalized_payload.get("meta", {}),
        "facts": normalized_payload.get("facts", []),
    }
    document_payload: dict[str, Any] = {"pages": [page_obj]}
    for key in ("images_dir", "metadata", "document_meta"):
        if key in normalized_payload:
            document_payload[key] = normalized_payload.get(key)
    return document_payload


def parse_import_json_text(text: str) -> ImportJsonParseResult:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("No JSON text was provided.")
    try:
        parsed = json.loads(stripped)
        return ImportJsonParseResult(
            payload=_normalize_top_level_import_payload(parsed),
            recovered=False,
            message=None,
        )
    except Exception:
        try:
            parsed = _parse_json_like(stripped)
            try:
                normalized_payload = _normalize_top_level_import_payload(parsed)
            except Exception:
                normalized_payload = None
            else:
                return ImportJsonParseResult(
                    payload=normalized_payload,
                    recovered=False,
                    message=None,
                )
        except Exception as exc:
            parsed = None

        pages_payload = _salvage_pages_payload_from_text(stripped)
        if pages_payload is not None:
            imported_pages = pages_payload.get("pages") if isinstance(pages_payload, dict) else None
            recovered_page_count = len(imported_pages) if isinstance(imported_pages, list) else 0
            recovered_fact_count = 0
            if isinstance(imported_pages, list):
                for page in imported_pages:
                    if isinstance(page, dict) and isinstance(page.get("facts"), list):
                        recovered_fact_count += len([fact for fact in page["facts"] if isinstance(fact, dict)])
            return ImportJsonParseResult(
                payload=pages_payload,
                recovered=True,
                message=(
                    "Recovered import from invalid JSON. "
                    f"Imported {recovered_page_count} page(s) and {recovered_fact_count} complete fact(s); "
                    "ignored trailing incomplete content."
                ),
            )
        page_sequence_payload = _salvage_page_sequence_payload_from_text(stripped)
        if page_sequence_payload is not None:
            imported_pages = page_sequence_payload.get("pages") if isinstance(page_sequence_payload, dict) else None
            recovered_page_count = len(imported_pages) if isinstance(imported_pages, list) else 0
            recovered_fact_count = 0
            if isinstance(imported_pages, list):
                for page in imported_pages:
                    if isinstance(page, dict) and isinstance(page.get("facts"), list):
                        recovered_fact_count += len([fact for fact in page["facts"] if isinstance(fact, dict)])
            return ImportJsonParseResult(
                payload=page_sequence_payload,
                recovered=True,
                message=(
                    "Recovered import from malformed page string sequence. "
                    f"Imported {recovered_page_count} page(s) and {recovered_fact_count} complete fact(s)."
                ),
            )
        page_payload = _salvage_single_page_payload_from_text(stripped)
        if page_payload is not None and isinstance(page_payload.get("facts"), list):
            return ImportJsonParseResult(
                payload=page_payload,
                recovered=True,
                message=(
                    "Recovered import from invalid JSON. "
                    f"Imported {len(page_payload['facts'])} complete fact(s); ignored trailing incomplete content."
                ),
            )
        raise ValueError("Could not parse the pasted JSON text.") from exc


_EDITABLE_META_TEXT_KEYS = frozenset({"entity_name", "title", "annotation_note"})
_EDITABLE_FACT_TEXT_KEYS = frozenset({"value", "comment_ref", "note_name", "note_ref"})
_HEBREW_CHAR_RE = re.compile(r"[\u0590-\u05FF]")


def _format_page_path(path_parts: Sequence[Any]) -> str:
    out: list[str] = []
    for part in path_parts:
        if isinstance(part, int):
            if not out:
                out.append(f"[{part}]")
            else:
                out[-1] = f"{out[-1]}[{part}]"
            continue
        text = str(part)
        out.append(text if not out else f".{text}")
    return "".join(out) or "$"


def _page_string_path_is_editable(path_parts: Sequence[Any]) -> bool:
    if len(path_parts) == 2 and path_parts[0] == "meta":
        return str(path_parts[1]) in _EDITABLE_META_TEXT_KEYS
    if len(path_parts) == 4 and path_parts[0] == "facts" and isinstance(path_parts[1], int):
        if path_parts[2] == "path" and isinstance(path_parts[3], int):
            return True
    if len(path_parts) == 3 and path_parts[0] == "facts" and isinstance(path_parts[1], int):
        return str(path_parts[2]) in _EDITABLE_FACT_TEXT_KEYS
    return False


def validate_page_text_correction(
    original_page: Any,
    corrected_page: Any,
) -> PageTextCorrectionValidationResult:
    changed_paths: list[str] = []
    changed_fields: list[dict[str, Any]] = []

    def _reject(reason: str) -> PageTextCorrectionValidationResult:
        return PageTextCorrectionValidationResult(
            accepted=False,
            changed_paths=tuple(changed_paths),
            changed_fields=tuple(changed_fields),
            rejection_reason=reason,
        )

    def _walk(original: Any, corrected: Any, path_parts: list[Any]) -> PageTextCorrectionValidationResult | None:
        if type(original) is not type(corrected):
            return _reject(
                f"Type changed at {_format_page_path(path_parts)}: "
                f"{type(original).__name__} -> {type(corrected).__name__}."
            )
        if isinstance(original, dict):
            if set(original.keys()) != set(corrected.keys()):
                return _reject(f"Object keys changed at {_format_page_path(path_parts)}.")
            for key in original.keys():
                failure = _walk(original.get(key), corrected.get(key), [*path_parts, str(key)])
                if failure is not None:
                    return failure
            return None
        if isinstance(original, list):
            if len(original) != len(corrected):
                return _reject(f"Array length changed at {_format_page_path(path_parts)}.")
            for index, (left, right) in enumerate(zip(original, corrected)):
                failure = _walk(left, right, [*path_parts, index])
                if failure is not None:
                    return failure
            return None
        if isinstance(original, str):
            if original == corrected:
                return None
            path_text = _format_page_path(path_parts)
            if not _page_string_path_is_editable(path_parts):
                return _reject(f"String changed at non-editable path {path_text}.")
            if not (_HEBREW_CHAR_RE.search(original) or _HEBREW_CHAR_RE.search(corrected)):
                return _reject(f"String changed without Hebrew text at {path_text}.")
            changed_paths.append(path_text)
            changed_fields.append(
                {
                    "path": path_text,
                    "original": original,
                    "corrected": corrected,
                }
            )
            return None
        if original != corrected:
            return _reject(f"Non-text value changed at {_format_page_path(path_parts)}.")
        return None

    failure = _walk(original_page, corrected_page, [])
    if failure is not None:
        return failure
    return PageTextCorrectionValidationResult(
        accepted=True,
        changed_paths=tuple(changed_paths),
        changed_fields=tuple(changed_fields),
        rejection_reason=None,
    )


def _clamp_bbox_to_image_bounds(
    bbox: Dict[str, Any],
    *,
    image_width: float,
    image_height: float,
) -> tuple[Dict[str, float], bool]:
    normalized = normalize_bbox_data(bbox)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)

    x = max(0.0, float(normalized["x"]))
    y = max(0.0, float(normalized["y"]))
    x = min(x, max(width - 1.0, 0.0))
    y = min(y, max(height - 1.0, 0.0))

    w = max(float(normalized["w"]), 1.0)
    h = max(float(normalized["h"]), 1.0)
    w = min(w, max(width - x, 1.0))
    h = min(h, max(height - y, 1.0))

    clamped = (
        x != float(normalized["x"])
        or y != float(normalized["y"])
        or w != float(normalized["w"])
        or h != float(normalized["h"])
    )
    return normalize_bbox_data({"x": x, "y": y, "w": w, "h": h}), clamped


def convert_imported_page_states(
    imported_states: Mapping[str, PageState],
    page_image_dimensions: Mapping[str, tuple[float, float] | None],
    *,
    bbox_mode: str = IMPORT_BBOX_MODE_ORIGINAL_PIXELS,
    max_pixels: int | None = None,
) -> tuple[Dict[str, PageState], ImportBBoxConversionStats]:
    allowed_modes = {
        IMPORT_BBOX_MODE_ORIGINAL_PIXELS,
        IMPORT_BBOX_MODE_NORMALIZED_1000,
        IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS,
    }
    normalized_mode = str(bbox_mode or "").strip().lower()
    if normalized_mode not in allowed_modes:
        raise ValueError(f"Unsupported import bbox mode: {bbox_mode}")
    if normalized_mode == IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS and (
        max_pixels is None or int(max_pixels) <= 0
    ):
        raise ValueError("max_pixels must be set to a positive integer for resized-image bbox import.")

    converted_states: Dict[str, PageState] = {}
    converted_count = 0
    clamped_count = 0
    skipped_count = 0

    for page_name, imported_state in imported_states.items():
        image_dims = page_image_dimensions.get(page_name)
        converted_facts: list[BoxRecord] = []
        for fact_index, record in enumerate(imported_state.facts):
            fact_data = dict(record.fact or {})
            use_placeholder_bbox = bool(fact_data.pop(_IMPORT_PLACEHOLDER_BBOX_FLAG, False))
            bbox = normalize_bbox_data(record.bbox) if not use_placeholder_bbox else make_placeholder_bbox(fact_index)
            clamped = False
            if use_placeholder_bbox and image_dims is not None:
                bbox, _placeholder_clamped = _clamp_bbox_to_image_bounds(
                    bbox,
                    image_width=image_dims[0],
                    image_height=image_dims[1],
                )
            if normalized_mode == IMPORT_BBOX_MODE_NORMALIZED_1000 and not use_placeholder_bbox:
                if image_dims is None:
                    skipped_count += 1
                else:
                    bbox = denormalize_bbox_from_1000(
                        bbox,
                        image_width=image_dims[0],
                        image_height=image_dims[1],
                    )
                    bbox, clamped = _clamp_bbox_to_image_bounds(
                        bbox,
                        image_width=image_dims[0],
                        image_height=image_dims[1],
                    )
                    converted_count += 1
            elif normalized_mode == IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS and not use_placeholder_bbox:
                if image_dims is None:
                    raise ValueError(f"Original image dimensions unavailable for imported page '{page_name}'.")
                bbox, clamped = restore_bbox_from_resized_pixels_with_stats(
                    bbox,
                    original_width=image_dims[0],
                    original_height=image_dims[1],
                    max_pixels=int(max_pixels or 0),
                )
                converted_count += 1

            if clamped:
                clamped_count += 1
            converted_facts.append(
                BoxRecord(
                    bbox=bbox,
                    fact=normalize_fact_data(fact_data),
                )
            )
        converted_states[str(page_name)] = PageState(
            meta=dict(imported_state.meta or {}),
            facts=converted_facts,
        )

    return converted_states, ImportBBoxConversionStats(
        converted=converted_count,
        clamped=clamped_count,
        skipped=skipped_count,
    )


def extract_document_meta(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "language": None,
            "reading_direction": None,
            "company_name": None,
            "company_id": None,
            "report_year": None,
            "entity_type": None,
        }
    return normalize_document_meta(payload.get("metadata", payload.get("document_meta")))


def build_annotations_payload(
    images_dir: Path,
    page_images: Sequence[Path],
    page_states: Dict[str, PageState],
    *,
    document_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload, _equation_findings = build_annotations_payload_with_findings(
        images_dir,
        page_images,
        page_states,
        document_meta=document_meta,
    )
    return save_canonical(payload)


def build_annotations_payload_with_findings(
    images_dir: Path,
    page_images: Sequence[Path],
    page_states: Dict[str, PageState],
    *,
    document_meta: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], list[dict[str, Any]]]:
    pages_out = []
    for idx, page_path in enumerate(page_images):
        page_name = page_path.name
        state = page_states.get(page_name, PageState(meta=default_page_meta(idx), facts=[]))
        meta = {**default_page_meta(idx), **(state.meta or {})}
        meta_model = PageMeta(**meta)

        facts_out = []
        normalized_facts = _assign_missing_fact_numbers([box.fact for box in state.facts])
        normalized_facts, _equation_findings = audit_and_rebuild_financial_facts(
            normalized_facts,
            statement_type=(meta_model.statement_type.value if meta_model.statement_type is not None else None),
            apply_repairs=True,
        )
        for box, normalized_fact in zip(state.facts, normalized_facts):
            fact_model = Fact(**normalized_fact)
            facts_out.append(
                {
                    "bbox": bbox_to_list(box.bbox),
                    **fact_model.model_dump(mode="json"),
                }
            )

        pages_out.append({"image": page_name, "meta": meta_model.model_dump(mode="json"), "facts": facts_out})

    payload: Dict[str, Any] = {"images_dir": str(images_dir), "pages": pages_out}
    compact_meta = compact_document_meta(document_meta)
    payload["metadata"] = compact_meta
    return canonicalize_with_findings(payload, strict_equation_guards=False)


def apply_entity_name_to_pages(
    page_states: Dict[str, PageState],
    page_images: Sequence[Path],
    entity_name: Optional[str],
    *,
    overwrite_existing: bool = False,
) -> int:
    normalized_entity = str(entity_name or "").strip()
    if not normalized_entity:
        return 0

    updated = 0
    for idx, page_path in enumerate(page_images):
        page_name = page_path.name
        state = page_states.get(page_name, PageState(meta=default_page_meta(idx), facts=[]))
        meta = {**default_page_meta(idx), **(state.meta or {})}
        existing_entity = str(meta.get("entity_name") or "").strip()
        if existing_entity and not overwrite_existing:
            continue
        if existing_entity == normalized_entity:
            continue
        meta["entity_name"] = normalized_entity
        page_states[page_name] = PageState(meta=meta, facts=list(state.facts))
        updated += 1
    return updated


def apply_entity_name_to_missing_pages(
    page_states: Dict[str, PageState],
    page_images: Sequence[Path],
    entity_name: Optional[str],
) -> int:
    return apply_entity_name_to_pages(
        page_states,
        page_images,
        entity_name,
        overwrite_existing=False,
    )


def serialize_annotations_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


__all__ = [
    "BoxRecord",
    "FACT_KEYS",
    "IMPORT_BBOX_MODE_NORMALIZED_1000",
    "IMPORT_BBOX_MODE_ORIGINAL_PIXELS",
    "IMPORT_BBOX_MODE_RESIZED_PIXELS_VIA_MAX_PIXELS",
    "ImportBBoxConversionStats",
    "ImportJsonParseResult",
    "CURRENCY_OPTIONS",
    "PageState",
    "SCALE_OPTIONS",
    "convert_imported_page_states",
    "build_annotations_payload",
    "bbox_to_list",
    "denormalize_bbox_from_1000",
    "default_fact_data",
    "default_page_meta",
    "extract_document_meta",
    "load_page_states",
    "normalize_import_payload_to_document",
    "parse_import_payload",
    "normalize_bbox_data",
    "normalize_fact_data",
    "parse_import_json_text",
    "PageTextCorrectionValidationResult",
    "validate_page_text_correction",
    "apply_entity_name_to_pages",
    "apply_entity_name_to_missing_pages",
    "serialize_annotations_json",
    "ValidationError",
]
