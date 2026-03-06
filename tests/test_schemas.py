from __future__ import annotations

import pytest
from pydantic import ValidationError

from finetree_annotator.schemas import DocumentMeta, ExtractedFact, PageExtraction


def _fact_payload(**overrides):
    payload = {
        "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
        "value": "10",
        "ref_comment": None,
        "note_flag": False,
        "note_num": None,
        "ref_note": None,
        "date": None,
        "path": [],
        "currency": None,
        "scale": None,
        "value_type": "amount",
    }
    payload.update(overrides)
    return payload


def test_extracted_fact_accepts_integer_note_num() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(note_flag=True, note_num=7))
    assert fact.note_num == 7


def test_extracted_fact_accepts_legacy_comment_alias() -> None:
    payload = _fact_payload()
    payload.pop("ref_comment")
    payload["comment"] = "legacy qualifier"
    fact = ExtractedFact.model_validate(payload)
    assert fact.ref_comment == "legacy qualifier"


def test_extracted_fact_accepts_legacy_note_reference_alias() -> None:
    payload = _fact_payload()
    payload.pop("ref_note")
    payload["note_reference"] = "legacy-ref"
    fact = ExtractedFact.model_validate(payload)
    assert fact.ref_note == "legacy-ref"


def test_extracted_fact_rejects_noninteger_note_num() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(note_flag=True, note_num="2ה׳"))


def test_extracted_fact_rejects_note_num_without_is_note() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(note_flag=False, note_num=7))


def test_extracted_fact_rejects_noncanonical_date() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(date="31.12.2024"))


def test_extracted_fact_rejects_empty_path_levels() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(path=["assets", "", "cash"]))


def test_page_extraction_rejects_note_facts_on_non_notes_page() -> None:
    with pytest.raises(ValidationError):
        PageExtraction.model_validate(
            {
                "meta": {"entity_name": None, "page_num": None, "type": "other", "title": None},
                "facts": [_fact_payload(note_flag=True, note_num=7)],
            }
        )


def test_document_meta_accepts_company_id_and_integer_report_year() -> None:
    meta = DocumentMeta.model_validate(
        {
            "language": "HE",
            "reading_direction": "RTL",
            "company_id": " cmp-1 ",
            "report_year": "2024",
        }
    )
    assert meta.language.value == "he"
    assert meta.reading_direction.value == "rtl"
    assert meta.company_id == "cmp-1"
    assert meta.report_year == 2024


def test_document_meta_rejects_noninteger_report_year() -> None:
    with pytest.raises(ValidationError):
        DocumentMeta.model_validate({"report_year": "FY2024"})
