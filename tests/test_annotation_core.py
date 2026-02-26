from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from finetree_annotator.annotation_core import (
    BoxRecord,
    PageState,
    build_annotations_payload,
    load_page_states,
    normalize_bbox_data,
    normalize_fact_data,
    propagate_entity_to_next_page,
    serialize_annotations_json,
)


def test_normalize_fact_data_coerces_path_currency_and_scale() -> None:
    fact = normalize_fact_data(
        {
            "value": "123",
            "path": ["assets", "", "cash", 2024],
            "currency": "ils",
            "scale": "1000",
            "value_type": "",
        }
    )

    assert fact["value"] == "123"
    assert fact["path"] == ["assets", "cash", "2024"]
    assert fact["currency"] == "ILS"
    assert fact["scale"] == 1000
    assert fact["value_type"] is None


def test_normalize_bbox_data_rounds_and_enforces_min_size() -> None:
    bbox = normalize_bbox_data({"x": "1.257", "y": 2.349, "w": 0.2, "h": "0"})
    assert bbox == {"x": 1.26, "y": 2.35, "w": 1.0, "h": 1.0}


def test_load_page_states_supports_flat_and_nested_fact_shapes() -> None:
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "notes", "entity_name": "Demo Co"},
                "facts": [
                    {
                        "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                        "value": "1,193",
                        "currency": "ils",
                        "path": ["assets", "cash"],
                        "scale": "1000",
                        "value_type": "amount",
                    },
                    {
                        "bbox": {"x": 1, "y": 2, "w": 0, "h": 5},
                        "fact": {
                            "value": "7%",
                            "path": ["ratios", "margin"],
                            "value_type": "%",
                        },
                    },
                ],
            },
            {"image": "page_9999.png", "meta": {"type": "other"}, "facts": []},
        ]
    }

    states = load_page_states(payload, ["page_0001.png", "page_0002.png"])
    assert set(states.keys()) == {"page_0001.png"}
    state = states["page_0001.png"]
    assert state.meta["entity_name"] == "Demo Co"
    assert len(state.facts) == 2
    assert state.facts[0].fact["currency"] == "ILS"
    assert state.facts[0].fact["scale"] == 1000
    assert state.facts[1].bbox["w"] == 1.0
    assert state.facts[1].fact["value_type"] == "%"


def test_build_annotations_payload_applies_defaults_for_missing_pages() -> None:
    page_images = [Path("page_0001.png"), Path("page_0002.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"entity_name": "ACME", "page_num": "1", "type": "notes", "title": "Notes"},
            facts=[
                BoxRecord(
                    bbox={"x": 10, "y": 20, "w": 30, "h": 40},
                    fact={
                        "value": "1,193",
                        "date": "2024-12-31",
                        "path": ["assets", "cash"],
                        "currency": "USD",
                        "scale": 1000,
                        "value_type": "amount",
                    },
                )
            ],
        )
    }

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    assert len(payload["pages"]) == 2
    page_1 = payload["pages"][0]
    page_2 = payload["pages"][1]
    assert page_1["meta"]["type"] == "notes"
    assert page_1["facts"][0]["currency"] == "USD"
    assert page_2["meta"]["type"] == "other"
    assert page_2["meta"]["page_num"] == "2"
    assert page_2["facts"] == []


def test_build_annotations_payload_raises_on_invalid_schema_values() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "not-a-real-page-type"},
            facts=[
                BoxRecord(
                    bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                    fact={"value": "100", "path": [], "value_type": "nope"},
                )
            ],
        )
    }

    with pytest.raises(ValidationError):
        build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)


def test_build_annotations_payload_rejects_currency_outside_allowed_list() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "other"},
            facts=[
                BoxRecord(
                    bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                    fact={"value": "100", "path": [], "currency": "CAD"},
                )
            ],
        )
    }

    with pytest.raises(ValidationError):
        build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)


def test_propagate_entity_to_next_page_sets_entity_name() -> None:
    page_images = [Path("page_0001.png"), Path("page_0002.png"), Path("page_0003.png")]
    page_states = {
        "page_0002.png": PageState(meta={"type": "notes", "title": "n2"}, facts=[]),
        "page_0003.png": PageState(meta={"entity_name": "Should stay"}, facts=[]),
    }

    propagate_entity_to_next_page(page_states, page_images, current_index=0, entity_name="ACME LTD")
    assert page_states["page_0002.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["title"] == "n2"

    propagate_entity_to_next_page(page_states, page_images, current_index=2, entity_name="LAST")
    assert page_states["page_0003.png"].meta["entity_name"] == "Should stay"


def test_serialize_annotations_json_keeps_hebrew_unescaped() -> None:
    payload = {
        "pages": [
            {
                "meta": {
                    "entity_name": "צלול - עמותה לאיכות הסביבה",
                }
            }
        ]
    }
    text = serialize_annotations_json(payload)
    assert "צלול" in text
    assert "\\u05e6" not in text
