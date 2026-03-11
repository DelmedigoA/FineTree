from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from finetree_annotator.annotation_core import (
    BoxRecord,
    PageState,
    apply_entity_name_to_missing_pages,
    bbox_to_list,
    build_annotations_payload,
    denormalize_bbox_from_1000,
    extract_document_meta,
    load_page_states,
    parse_import_payload,
    normalize_bbox_data,
    normalize_fact_data,
    serialize_annotations_json,
)


def test_normalize_fact_data_coerces_path_currency_and_scale() -> None:
    fact = normalize_fact_data(
        {
            "value": "123",
            "note": "  *without debt insurance  ",
            "is_beur": "true",
            "beur_num": " 2ה׳ ",
            "refference": "  ref-001  ",
            "path": ["assets", "", "cash", 2024],
            "currency": "ils",
            "scale": "1000",
            "value_type": "",
        }
    )

    assert fact["value"] == "123"
    assert fact["comment_ref"] == "*without debt insurance"
    assert fact["note_flag"] is True
    assert fact["note_num"] is None
    assert fact["note_ref"] == "ref-001"
    assert fact["path"] == ["assets", "cash", "2024"]
    assert fact["currency"] == "ILS"
    assert fact["scale"] == 1000
    assert fact["value_type"] is None
    assert fact["natural_sign"] == "positive"


def test_normalize_bbox_data_rounds_and_enforces_min_size() -> None:
    bbox = normalize_bbox_data({"x": "1.257", "y": 2.349, "w": 0.2, "h": "0"})
    assert bbox == {"x": 1.26, "y": 2.35, "w": 1.0, "h": 1.0}


def test_normalize_bbox_data_supports_array_shape() -> None:
    bbox = normalize_bbox_data([1.257, 2.349, 0.2, 0])
    assert bbox == {"x": 1.26, "y": 2.35, "w": 1.0, "h": 1.0}
    assert bbox_to_list(bbox) == [1.26, 2.35, 1.0, 1.0]


def test_denormalize_bbox_from_1000_scales_to_image_pixels() -> None:
    bbox = denormalize_bbox_from_1000({"x": 250, "y": 500, "w": 100, "h": 50}, image_width=2000, image_height=3000)
    assert bbox == {"x": 500.0, "y": 1500.0, "w": 200.0, "h": 150.0}


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
                        "note": "*estimated",
                        "is_beur": True,
                        "beur_num": "5",
                        "refference": "A1",
                        "currency": "ils",
                        "path": ["assets", "cash"],
                        "scale": "1000",
                        "value_type": "amount",
                    },
                    {
                        "bbox": [1, 2, 0, 5],
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
    assert state.meta["page_type"] == "statements"
    assert state.meta["statement_type"] == "notes_to_financial_statements"
    assert len(state.facts) == 2
    assert state.facts[0].fact["currency"] == "ILS"
    assert state.facts[0].fact["comment_ref"] == "*estimated"
    assert state.facts[0].fact["note_flag"] is True
    assert state.facts[0].fact["note_num"] == 5
    assert state.facts[0].fact["note_ref"] == "A1"
    assert state.facts[0].fact["scale"] == 1000
    assert state.facts[1].bbox["w"] == 1.0
    assert state.facts[1].fact["value_type"] == "percent"


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
                        "equation": "1,000 + 193",
                        "note": "*estimated",
                        "is_beur": True,
                        "beur_num": "5",
                        "refference": "row-12",
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
    assert page_1["meta"]["page_type"] == "statements"
    assert page_1["meta"]["statement_type"] == "notes_to_financial_statements"
    assert page_1["facts"][0]["currency"] == "USD"
    assert page_1["facts"][0]["comment_ref"] == "*estimated"
    assert page_1["facts"][0]["fact_num"] == 1
    assert page_1["facts"][0]["equations"] == [{"equation": "1,000 + 193", "fact_equation": None}]
    assert page_1["facts"][0]["note_flag"] is True
    assert page_1["facts"][0]["note_num"] == 5
    assert page_1["facts"][0]["note_ref"] == "row-12"
    assert page_1["facts"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert page_2["meta"]["page_type"] == "other"
    assert page_2["meta"]["statement_type"] is None
    assert page_2["meta"]["page_num"] is None
    assert page_2["facts"] == []


def test_build_annotations_payload_raises_on_invalid_schema_values() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"page_type": "not-a-real-page-type"},
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


def test_build_annotations_payload_accepts_activities_page_type() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "activities", "page_num": "7"},
            facts=[],
        )
    }

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    assert payload["pages"][0]["meta"]["page_type"] == "statements"
    assert payload["pages"][0]["meta"]["statement_type"] == "statement_of_activities"


def test_build_annotations_payload_normalizes_currency_outside_allowed_list_to_none() -> None:
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

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    assert payload["pages"][0]["facts"][0]["currency"] is None


def test_apply_entity_name_to_missing_pages_sets_only_absent_values() -> None:
    page_images = [Path("page_0001.png"), Path("page_0002.png"), Path("page_0003.png")]
    page_states = {
        "page_0001.png": PageState(meta={"entity_name": "Current Page"}, facts=[]),
        "page_0002.png": PageState(meta={"type": "notes", "title": "n2"}, facts=[]),
        "page_0003.png": PageState(meta={"entity_name": "Should stay"}, facts=[]),
    }

    updated = apply_entity_name_to_missing_pages(page_states, page_images, entity_name="ACME LTD")
    assert updated == 1
    assert page_states["page_0002.png"].meta["entity_name"] == "ACME LTD"
    assert page_states["page_0002.png"].meta["title"] == "n2"
    assert page_states["page_0003.png"].meta["entity_name"] == "Should stay"

    no_change = apply_entity_name_to_missing_pages(page_states, page_images, entity_name="LAST")
    assert no_change == 0


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


def test_parse_import_payload_supports_single_page_shape_without_image() -> None:
    payload = {
        "meta": {"type": "profits", "page_num": "4"},
        "facts": [{"bbox": [1, 2, 3, 4], "value": "10", "refference": "5", "path": []}],
    }
    states = parse_import_payload(payload, ["page_0001.png", "page_0002.png"], "page_0002.png")
    assert set(states.keys()) == {"page_0002.png"}
    assert states["page_0002.png"].meta["page_type"] == "statements"
    assert states["page_0002.png"].meta["statement_type"] == "income_statement"
    assert states["page_0002.png"].facts[0].fact["note_ref"] == "5"


def test_load_page_states_preserves_equation_field() -> None:
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other", "annotation_note": "revisit", "annotation_status": "approved"},
                "facts": [
                    {
                        "bbox": [1, 2, 3, 4],
                        "value": "10",
                        "fact_num": 4,
                        "equation": "7 + 3",
                        "fact_equation": "f1 + f3",
                        "path": [],
                    }
                ],
            }
        ]
    }
    states = load_page_states(payload, ["page_0001.png"])
    assert states["page_0001.png"].meta["annotation_note"] == "revisit"
    assert states["page_0001.png"].meta["annotation_status"] == "approved"
    assert states["page_0001.png"].facts[0].fact["fact_num"] == 1
    assert states["page_0001.png"].facts[0].fact["equations"] == [{"equation": "7 + 3", "fact_equation": "f1 + f3"}]


def test_build_annotations_payload_backfills_missing_fact_num_for_legacy_facts() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "other"},
            facts=[
                BoxRecord(
                    bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                    fact={"value": "10", "path": []},
                ),
                BoxRecord(
                    bbox={"x": 20, "y": 0, "w": 10, "h": 10},
                    fact={"value": "20", "path": []},
                ),
            ],
        )
    }

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    assert payload["pages"][0]["meta"]["annotation_note"] is None
    assert payload["pages"][0]["meta"]["annotation_status"] is None
    assert [fact["fact_num"] for fact in payload["pages"][0]["facts"]] == [1, 2]


def test_build_annotations_payload_resequences_fact_nums_contiguously() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "other"},
            facts=[
                BoxRecord(
                    bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                    fact={"value": "10", "fact_num": 3, "path": []},
                ),
                BoxRecord(
                    bbox={"x": 20, "y": 0, "w": 10, "h": 10},
                    fact={"value": "20", "fact_num": 3, "path": []},
                ),
                BoxRecord(
                    bbox={"x": 40, "y": 0, "w": 10, "h": 10},
                    fact={"value": "30", "fact_num": 7, "path": []},
                ),
            ],
        )
    }

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    assert [fact["fact_num"] for fact in payload["pages"][0]["facts"]] == [1, 2, 3]


def test_build_annotations_payload_remaps_fact_equation_refs_when_fact_nums_shift() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"type": "other"},
            facts=[
                BoxRecord(
                    bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                    fact={"value": "900", "fact_num": 4, "path": []},
                ),
                BoxRecord(
                    bbox={"x": 20, "y": 0, "w": 10, "h": 10},
                    fact={"value": "100", "fact_num": 1, "path": []},
                ),
                BoxRecord(
                    bbox={"x": 40, "y": 0, "w": 10, "h": 10},
                    fact={"value": "20", "fact_num": 2, "path": []},
                ),
                BoxRecord(
                    bbox={"x": 60, "y": 0, "w": 10, "h": 10},
                    fact={
                        "value": "120",
                        "fact_num": 3,
                        "equation": "100 + 20",
                        "fact_equation": "f1 + f2",
                        "path": [],
                    },
                ),
            ],
        )
    }

    payload = build_annotations_payload(Path("data/pdf_images/test"), page_images, page_states)
    facts = payload["pages"][0]["facts"]
    assert [fact["fact_num"] for fact in facts] == [1, 2, 3, 4]
    assert facts[3]["equations"][0]["fact_equation"] == "f2 + f3"
    assert facts[3]["equations"][0]["equation"] == "100 + 20"


def test_parse_import_payload_supports_full_document_shape() -> None:
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "notes"},
                "facts": [{"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "value": "10", "refference": "", "path": []}],
            },
            {
                "image": "page_9999.png",
                "meta": {"type": "other"},
                "facts": [],
            },
        ]
    }
    states = parse_import_payload(payload, ["page_0001.png"], "page_0001.png")
    assert set(states.keys()) == {"page_0001.png"}
    assert states["page_0001.png"].meta["page_type"] == "statements"
    assert states["page_0001.png"].meta["statement_type"] == "notes_to_financial_statements"


def test_extract_document_meta_normalizes_supported_values() -> None:
    payload = {"document_meta": {"language": "HE", "reading_direction": "RTL", "company_id": " 1234 ", "report_year": "2024"}}
    doc_meta = extract_document_meta(payload)
    assert doc_meta == {
        "language": "he",
        "reading_direction": "rtl",
        "company_name": None,
        "company_id": "1234",
        "report_year": 2024,
        "report_scope": None,
        "entity_type": None,
    }


def test_build_annotations_payload_includes_document_meta_when_present() -> None:
    page_images = [Path("page_0001.png")]
    page_states = {
        "page_0001.png": PageState(
            meta={"entity_name": "ACME", "page_num": "1", "type": "other", "title": "Cover"},
            facts=[],
        )
    }
    payload = build_annotations_payload(
        Path("data/pdf_images/test"),
        page_images,
        page_states,
        document_meta={"language": "en", "reading_direction": "ltr", "company_id": "cmp-7", "report_year": 2025},
    )
    assert payload["metadata"] == {
        "language": "en",
        "reading_direction": "ltr",
        "company_id": "cmp-7",
        "report_year": 2025,
    }
