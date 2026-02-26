"""Core data helpers for the PDF annotator.

This module intentionally has no Qt dependency so it can be unit tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import ValidationError

from .schemas import Currency, Fact, PageMeta, PageType, Scale

FACT_KEYS = ("value", "date", "path", "currency", "scale", "value_type")
CURRENCY_OPTIONS = [c.value for c in Currency]
SCALE_OPTIONS = [s.value for s in Scale]


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
        "date": None,
        "path": [],
        "currency": None,
        "scale": None,
        "value_type": None,
    }


def normalize_fact_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = default_fact_data()
    if not data:
        return base

    merged = {**base, **data}
    if not isinstance(merged.get("path"), list):
        merged["path"] = []
    merged["path"] = [str(x).strip() for x in merged["path"] if str(x).strip()]

    if merged.get("scale") in ("", None):
        merged["scale"] = None
    elif isinstance(merged["scale"], str):
        try:
            merged["scale"] = int(merged["scale"])
        except ValueError:
            merged["scale"] = None

    if merged.get("currency") in ("", None):
        merged["currency"] = None
    elif isinstance(merged["currency"], str):
        currency = merged["currency"].strip().upper()
        merged["currency"] = currency or None

    if merged.get("value_type") == "":
        merged["value_type"] = None

    return merged


def normalize_bbox_data(data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw = data or {}
    x = float(raw.get("x", 0.0))
    y = float(raw.get("y", 0.0))
    w = max(float(raw.get("w", 1.0)), 1.0)
    h = max(float(raw.get("h", 1.0)), 1.0)
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def default_page_meta(index: int) -> Dict[str, Any]:
    return {
        "entity_name": None,
        "page_num": str(index + 1),
        "type": PageType.other.value,
        "title": None,
    }


def _coerce_raw_fact(raw_fact: Dict[str, Any]) -> Dict[str, Any]:
    if "fact" in raw_fact and isinstance(raw_fact["fact"], dict):
        return normalize_fact_data(raw_fact["fact"])
    return normalize_fact_data({k: raw_fact.get(k) for k in FACT_KEYS})


def load_page_states(payload: Dict[str, Any], page_image_names: Iterable[str]) -> Dict[str, PageState]:
    available = set(page_image_names)
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
        raw_facts = page.get("facts", [])
        facts: List[BoxRecord] = []
        if isinstance(raw_facts, list):
            for raw_fact in raw_facts:
                if not isinstance(raw_fact, dict):
                    continue
                bbox = normalize_bbox_data(raw_fact.get("bbox") if isinstance(raw_fact.get("bbox"), dict) else None)
                facts.append(BoxRecord(bbox=bbox, fact=_coerce_raw_fact(raw_fact)))
        states[str(page_name)] = PageState(meta=meta, facts=facts)
    return states


def build_annotations_payload(
    images_dir: Path,
    page_images: Sequence[Path],
    page_states: Dict[str, PageState],
) -> Dict[str, Any]:
    pages_out = []
    for idx, page_path in enumerate(page_images):
        page_name = page_path.name
        state = page_states.get(page_name, PageState(meta=default_page_meta(idx), facts=[]))
        meta = {**default_page_meta(idx), **(state.meta or {})}
        meta_model = PageMeta(**meta)

        facts_out = []
        for box in state.facts:
            fact_model = Fact(**normalize_fact_data(box.fact))
            facts_out.append(
                {
                    "bbox": normalize_bbox_data(box.bbox),
                    **fact_model.model_dump(mode="json"),
                }
            )

        pages_out.append({"image": page_name, "meta": meta_model.model_dump(mode="json"), "facts": facts_out})

    return {"images_dir": str(images_dir), "pages": pages_out}


def propagate_entity_to_next_page(
    page_states: Dict[str, PageState],
    page_images: Sequence[Path],
    current_index: int,
    entity_name: Optional[str],
) -> None:
    next_index = current_index + 1
    if next_index >= len(page_images):
        return

    next_name = page_images[next_index].name
    next_state = page_states.get(next_name, PageState(meta=default_page_meta(next_index), facts=[]))
    next_meta = {**default_page_meta(next_index), **(next_state.meta or {})}
    next_meta["entity_name"] = entity_name
    page_states[next_name] = PageState(meta=next_meta, facts=list(next_state.facts))


def serialize_annotations_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


__all__ = [
    "BoxRecord",
    "FACT_KEYS",
    "CURRENCY_OPTIONS",
    "PageState",
    "SCALE_OPTIONS",
    "build_annotations_payload",
    "default_fact_data",
    "default_page_meta",
    "load_page_states",
    "normalize_bbox_data",
    "normalize_fact_data",
    "propagate_entity_to_next_page",
    "serialize_annotations_json",
    "ValidationError",
]
