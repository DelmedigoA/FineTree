from __future__ import annotations

import pytest
from pydantic import ValidationError

from finetree_annotator.schemas import DocumentMeta, ExtractedFact, PageExtraction, PageMeta


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


def test_extracted_fact_accepts_points_value_type() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value_type="points"))
    assert fact.value_type is not None
    assert fact.value_type.value == "points"


def test_extracted_fact_accepts_value_context() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value_context="tabular"))
    assert fact.value_context is not None
    assert fact.value_context.value == "tabular"


def test_extracted_fact_rejects_deprecated_balance_type() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(balance_type="debit"))


def test_extracted_fact_rejects_deprecated_aggregation_role() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(aggregation_role="subtractive"))


def test_extracted_fact_accepts_row_role() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(row_role="total"))
    assert fact.row_role is not None
    assert fact.row_role.value == "total"


def test_extracted_fact_accepts_total_without_fact_equation() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(row_role="total"))
    assert fact.row_role is not None
    assert fact.row_role.value == "total"


def test_extracted_fact_accepts_recurrent_duration_type() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(duration_type="recurrent"))
    assert fact.duration_type is not None
    assert fact.duration_type.value == "recurrent"


def test_extracted_fact_maps_legacy_recurring_duration_type_to_recurrent() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(duration_type="recurring"))
    assert fact.duration_type is not None
    assert fact.duration_type.value == "recurrent"


def test_extracted_fact_derives_natural_sign_from_value() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value="(123)", natural_sign="positive"))
    assert fact.natural_sign is not None
    assert fact.natural_sign.value == "negative"


def test_extracted_fact_derives_natural_sign_from_angle_bracketed_negative_value() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value="<-123>", natural_sign="positive"))
    assert fact.natural_sign is not None
    assert fact.natural_sign.value == "negative"


def test_extracted_fact_sets_natural_sign_null_for_dash_value() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value="-", natural_sign="positive"))
    assert fact.natural_sign is None


def test_extracted_fact_accepts_symbol_rich_value_text() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(value="<>)().*, 10"))
    assert fact.value == "<>)().*, 10"


def test_extracted_fact_accepts_equation() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(equation="100 - 20 + 5"))
    assert fact.equations is not None
    assert fact.equations[0].equation == "100 - 20 + 5"


def test_extracted_fact_accepts_equations_list_and_syncs_active_equation() -> None:
    fact = ExtractedFact.model_validate(
        _fact_payload(
            equation="80 + 40",
            fact_equation="f1 + f2",
            equations=[
                {"equation": "100 + 20", "fact_equation": "f9 + f10"},
                {"equation": "80 + 40", "fact_equation": "f1 + f2"},
            ],
        )
    )
    assert fact.equations is not None
    assert fact.equations[0].equation == "80 + 40"
    assert fact.equations[0].fact_equation == "f1 + f2"
    assert fact.equations[1].equation == "100 + 20"


def test_extracted_fact_accepts_fact_num_and_fact_equation() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(fact_num="7", equation="10", fact_equation="f1 + f4 - f6"))
    assert fact.fact_num == 7
    assert fact.equations is not None
    assert fact.equations[0].fact_equation == "f1 + f4 - f6"


def test_extracted_fact_rejects_empty_or_null_value() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(value=""))
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(value=None))


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


def test_extracted_fact_allows_note_num_without_is_note() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(note_flag=False, note_num=7))
    assert fact.note_flag is False
    assert fact.note_num == 7


def test_extracted_fact_rejects_noncanonical_date() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(date="31.12.2024"))


def test_extracted_fact_accepts_duration_without_range() -> None:
    fact = ExtractedFact.model_validate(_fact_payload(period_type="duration", period_start=None, period_end=None))
    assert fact.period_type is not None
    assert fact.period_type.value == "duration"
    assert fact.period_start is None
    assert fact.period_end is None


def test_extracted_fact_accepts_duration_with_complete_range() -> None:
    fact = ExtractedFact.model_validate(
        _fact_payload(period_type="duration", period_start="2024-01-01", period_end="2024-12-31")
    )
    assert fact.period_type is not None
    assert fact.period_start == "2024-01-01"
    assert fact.period_end == "2024-12-31"


def test_extracted_fact_rejects_duration_with_one_missing_boundary() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(period_type="duration", period_start="2024-01-01", period_end=None))
    with pytest.raises(ValidationError):
        ExtractedFact.model_validate(_fact_payload(period_type="duration", period_start=None, period_end="2024-12-31"))


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


def test_document_meta_accepts_report_scope() -> None:
    meta = DocumentMeta.model_validate({"report_scope": "consolidated"})
    assert meta.report_scope is not None
    assert meta.report_scope.value == "consolidated"


def test_document_meta_accepts_null_string_report_scope() -> None:
    meta = DocumentMeta.model_validate({"report_scope": "null"})
    assert meta.report_scope is None


def test_document_meta_rejects_noninteger_report_year() -> None:
    with pytest.raises(ValidationError):
        DocumentMeta.model_validate({"report_year": "FY2024"})


def test_page_meta_accepts_other_declaration_statement_type() -> None:
    meta = PageMeta.model_validate(
        {
            "entity_name": None,
            "page_num": None,
            "page_type": "statements",
            "statement_type": "other_declaration",
            "title": None,
        }
    )
    assert meta.statement_type is not None
    assert meta.statement_type.value == "other_declaration"


def test_page_meta_accepts_annotation_note() -> None:
    meta = PageMeta.model_validate({"annotation_note": "  revisit this page  "})
    assert meta.annotation_note == "revisit this page"


def test_page_meta_accepts_annotation_status() -> None:
    meta = PageMeta.model_validate({"annotation_status": " flag "})
    assert meta.annotation_status is not None
    assert meta.annotation_status.value == "flagged"
