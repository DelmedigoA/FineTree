"""Core data helpers for the PDF annotator.

This module intentionally has no Qt dependency so it can be unit tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import ValidationError

from .equation_integrity import audit_and_rebuild_financial_facts
from .equation_integrity import resequence_fact_numbers_and_remap_fact_equations
from .fact_ordering import compact_document_meta, normalize_document_meta
from .fact_normalization import normalize_fact_payload
from .schema_io import save_canonical
from .schema_contract import CANONICAL_FACT_KEYS, CURRENCY_VALUES, SCALE_VALUES
from .schemas import Fact, PageMeta, PageType

FACT_KEYS = tuple(CANONICAL_FACT_KEYS)
CURRENCY_OPTIONS = list(CURRENCY_VALUES)
SCALE_OPTIONS = list(SCALE_VALUES)


@dataclass
class BoxRecord:
    bbox: Dict[str, float]
    fact: Dict[str, Any]


@dataclass
class PageState:
    meta: Dict[str, Any] = field(default_factory=dict)
    facts: List[BoxRecord] = field(default_factory=list)


def default_fact_data() -> Dict[str, Any]:
    return {
        "value": "",
        "fact_num": None,
        "equation": None,
        "fact_equation": None,
        "balance_type": None,
        "natural_sign": None,
        "row_role": "detail",
        "aggregation_role": "additive",
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


def normalize_bbox_data(data: Optional[Any]) -> Dict[str, float]:
    if isinstance(data, dict):
        x_raw = data.get("x", 0.0)
        y_raw = data.get("y", 0.0)
        w_raw = data.get("w", 1.0)
        h_raw = data.get("h", 1.0)
    elif isinstance(data, (list, tuple)) and len(data) >= 4:
        x_raw, y_raw, w_raw, h_raw = data[0], data[1], data[2], data[3]
    else:
        x_raw, y_raw, w_raw, h_raw = 0.0, 0.0, 1.0, 1.0

    x = float(x_raw)
    y = float(y_raw)
    w = max(float(w_raw), 1.0)
    h = max(float(h_raw), 1.0)
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def bbox_to_list(data: Optional[Any]) -> List[float]:
    bbox = normalize_bbox_data(data)
    return [bbox["x"], bbox["y"], bbox["w"], bbox["h"]]


def denormalize_bbox_from_1000(data: Optional[Dict[str, Any]], image_width: float, image_height: float) -> Dict[str, float]:
    bbox = normalize_bbox_data(data)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    return normalize_bbox_data(
        {
            "x": (bbox["x"] * width) / 1000.0,
            "y": (bbox["y"] * height) / 1000.0,
            "w": (bbox["w"] * width) / 1000.0,
            "h": (bbox["h"] * height) / 1000.0,
        }
    )


def default_page_meta(index: int) -> Dict[str, Any]:
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


def load_page_states(payload: Dict[str, Any], page_image_names: Iterable[str]) -> Dict[str, PageState]:
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
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        page_index = index_map.get(str(page_name), 0)
        meta_model = PageMeta(**{**default_page_meta(page_index), **meta})
        raw_facts = page.get("facts", [])
        facts: List[BoxRecord] = []
        if isinstance(raw_facts, list):
            raw_fact_dicts = [fact for fact in raw_facts if isinstance(fact, dict)]
            normalized_facts = _assign_missing_fact_numbers([_coerce_raw_fact(fact) for fact in raw_fact_dicts])
            for raw_fact, normalized_fact in zip(raw_fact_dicts, normalized_facts):
                bbox = normalize_bbox_data(raw_fact.get("bbox"))
                facts.append(BoxRecord(bbox=bbox, fact=normalized_fact))
        states[str(page_name)] = PageState(meta=meta_model.model_dump(mode="json"), facts=facts)
    return states


def parse_import_payload(
    payload: Any,
    page_image_names: Sequence[str],
    default_page_image_name: Optional[str],
) -> Dict[str, PageState]:
    if isinstance(payload, list):
        payload = {"meta": {}, "facts": payload}
    if not isinstance(payload, dict):
        return {}

    if isinstance(payload.get("pages"), list):
        return load_page_states(payload, page_image_names)

    default_image = default_page_image_name or (page_image_names[0] if page_image_names else None)
    if not default_image:
        return {}

    image_name = payload.get("image")
    if not isinstance(image_name, str) or not image_name.strip():
        image_name = default_image

    page_obj = {
        "image": image_name,
        "meta": payload.get("meta", {}),
        "facts": payload.get("facts", []),
    }
    return load_page_states({"pages": [page_obj]}, page_image_names)


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
    return save_canonical(payload)


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
    "CURRENCY_OPTIONS",
    "PageState",
    "SCALE_OPTIONS",
    "build_annotations_payload",
    "bbox_to_list",
    "denormalize_bbox_from_1000",
    "default_fact_data",
    "default_page_meta",
    "extract_document_meta",
    "load_page_states",
    "parse_import_payload",
    "normalize_bbox_data",
    "normalize_fact_data",
    "apply_entity_name_to_pages",
    "apply_entity_name_to_missing_pages",
    "serialize_annotations_json",
    "ValidationError",
]
