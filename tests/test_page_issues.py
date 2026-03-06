from __future__ import annotations

from finetree_annotator.annotation_core import BoxRecord, PageState
from finetree_annotator.page_issues import (
    aggregate_document_issues,
    validate_document_issues,
    validate_page_issues,
)


def _fact(**overrides):
    base = {
        "value": "10",
        "comment": None,
        "note_flag": False,
        "note_num": None,
        "note_reference": None,
        "date": None,
        "path": [],
        "currency": None,
        "scale": None,
        "value_type": None,
    }
    base.update(overrides)
    return BoxRecord(bbox={"x": 1, "y": 2, "w": 3, "h": 4}, fact=base)


def test_validate_page_issues_detects_notes_page_without_note_facts() -> None:
    summary = validate_page_issues(
        "page_0007.png",
        PageState(
            meta={"type": "notes"},
            facts=[
                _fact(value="12", note_flag=False),
                _fact(value="13", note_flag=False, note_num=None),
            ],
        ),
    )
    codes = [issue.code for issue in summary.issues]
    assert "notes_page_missing_note_flag" in codes
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 1


def test_validate_page_issues_detects_fact_level_reg_flags() -> None:
    summary = validate_page_issues(
        "page_0007.png",
        PageState(
            meta={"type": "notes"},
            facts=[
                _fact(value="", note_flag=False),
                _fact(value="12", note_flag=True, note_num=None),
                _fact(value="13", note_flag=False, note_num=7),
            ],
        ),
    )
    codes = [issue.code for issue in summary.issues]
    assert "fact_missing_value" in codes
    assert "note_flag_missing_note_num" in codes
    assert "note_num_without_note_flag" in codes
    assert summary.reg_flag_count == 2
    assert summary.warning_count == 1


def test_validate_page_issues_detects_non_notes_is_note_and_mixed_fields() -> None:
    summary = validate_page_issues(
        "page_0008.png",
        PageState(
            meta={"type": "other"},
            facts=[
                _fact(value="10", note_flag=True, note_num=5, currency="USD", scale=1, value_type="amount"),
                _fact(value="11", note_flag=False, currency="ILS", scale=1000, value_type="%"),
            ],
        ),
    )
    codes = [issue.code for issue in summary.issues]
    assert "note_flag_on_non_notes_page" in codes
    assert "mixed_currency" in codes
    assert "mixed_scale" in codes
    assert "mixed_value_type" in codes
    assert summary.reg_flag_count == 1
    assert "scale_on_percent_fact" in codes
    assert "percent_with_currency" in codes
    assert summary.warning_count == 8
    warning_fact_indexes = sorted(
        issue.fact_index
        for issue in summary.issues
        if issue.severity == "warning"
    )
    assert warning_fact_indexes == [0, 0, 0, 1, 1, 1, 1, 1]


def test_validate_page_issues_ignores_empty_warning_values() -> None:
    summary = validate_page_issues(
        "page_0009.png",
        PageState(
            meta={"type": "other"},
            facts=[
                _fact(value="10", currency="USD", scale=None, value_type="amount"),
                _fact(value="11", currency="", scale=None, value_type=""),
                _fact(value="12", currency="USD", scale=None, value_type="amount"),
            ],
        ),
    )
    assert summary.warning_count == 0


def test_validate_page_issues_points_to_outlier_when_majority_exists() -> None:
    summary = validate_page_issues(
        "page_0010.png",
        PageState(
            meta={"type": "other"},
            facts=[
                _fact(value="10", scale=1000),
                _fact(value="11", scale=1000),
                _fact(value="12", scale=1),
            ],
        ),
    )
    warnings = [issue for issue in summary.issues if issue.severity == "warning"]
    assert summary.warning_count == 1
    assert len(warnings) == 1
    assert warnings[0].code == "mixed_scale"
    assert warnings[0].fact_index == 2
    assert "most facts use '1000'" in warnings[0].message


def test_validate_page_issues_detects_selected_new_fact_rules() -> None:
    summary = validate_page_issues(
        "page_0011.png",
        PageState(
            meta={"type": "notes"},
            facts=[
                _fact(
                    value="12%",
                    note_flag=True,
                    note_num=7,
                    note_reference="n7",
                    date="2024-13-40",
                    path=["assets", "", "cash"],
                    currency="USD",
                    scale=1000,
                    value_type="amount",
                )
            ],
        ),
    )
    codes = {issue.code for issue in summary.issues}
    assert "note_reference_on_notes_page" in codes
    assert "invalid_date" in codes
    assert "path_contains_empty_level" in codes
    assert "amount_type_percent_value" in codes


def test_validate_page_issues_warns_when_path_contains_note_name() -> None:
    summary = validate_page_issues(
        "page_0011d.png",
        PageState(
            meta={"type": "notes"},
            facts=[
                _fact(
                    value="12",
                    note_flag=True,
                    note_num=8,
                    note_name="ביאור 8 - רכוש קבוע",
                    path=["מאזן", "פירוט ביאור 8 - רכוש קבוע"],
                )
            ],
        ),
    )
    overlap_issues = [issue for issue in summary.issues if issue.code == "path_overlaps_note_name"]
    assert len(overlap_issues) == 1
    assert overlap_issues[0].field_name == "path"
    assert "contains note_name" in overlap_issues[0].message


def test_validate_page_issues_warns_on_noninteger_note_num() -> None:
    summary = validate_page_issues(
        "page_0011c.png",
        PageState(
            meta={"type": "notes"},
            facts=[
                _fact(value="12", note_flag=True, note_num="2ה׳"),
            ],
        ),
    )
    codes = {issue.code for issue in summary.issues}
    assert "noninteger_note_num" in codes
    assert "note_flag_missing_note_num" not in codes


def test_validate_page_issues_accepts_year_month_date() -> None:
    summary = validate_page_issues(
        "page_0011b.png",
        PageState(
            meta={"type": "other"},
            facts=[
                _fact(
                    value="12",
                    date="2024-09",
                )
            ],
        ),
    )
    codes = {issue.code for issue in summary.issues}
    assert "invalid_date" not in codes


def test_validate_page_issues_detects_percent_specific_warnings() -> None:
    summary = validate_page_issues(
        "page_0012.png",
        PageState(
            meta={"type": "other"},
            facts=[
                _fact(value="8", value_type="%", currency="USD", scale=1000),
            ],
        ),
    )
    codes = [issue.code for issue in summary.issues]
    assert "scale_on_percent_fact" in codes
    assert "percent_with_currency" in codes
    assert summary.reg_flag_count == 0
    assert summary.warning_count == 2


def test_validate_document_issues_detects_missing_entity_and_duplicate_page_num() -> None:
    doc_summary = validate_document_issues(
        [
            ("page_0001.png", PageState(meta={"entity_name": "Acme Ltd", "page_num": "5"}, facts=[])),
            ("page_0002.png", PageState(meta={"entity_name": "", "page_num": "5"}, facts=[])),
            ("page_0003.png", PageState(meta={"entity_name": "Acme Ltd", "page_num": "7"}, facts=[])),
        ]
    )
    page_two_codes = {issue.code for issue in doc_summary.page_summaries["page_0002.png"].issues}
    assert "missing_entity_name_among_majority" in page_two_codes
    assert "duplicate_page_num" in page_two_codes
    assert "missing_page_num_between_numbered_pages" not in page_two_codes


def test_validate_document_issues_detects_page_num_sequence_problems() -> None:
    doc_summary = validate_document_issues(
        [
            ("page_0001.png", PageState(meta={"page_num": "1"}, facts=[])),
            ("page_0002.png", PageState(meta={"page_num": "3"}, facts=[])),
            ("page_0003.png", PageState(meta={"page_num": "2"}, facts=[])),
            ("page_0004.png", PageState(meta={"page_num": ""}, facts=[])),
            ("page_0005.png", PageState(meta={"page_num": "4"}, facts=[])),
        ]
    )
    page_two_codes = {issue.code for issue in doc_summary.page_summaries["page_0002.png"].issues}
    page_three_codes = {issue.code for issue in doc_summary.page_summaries["page_0003.png"].issues}
    page_four_codes = {issue.code for issue in doc_summary.page_summaries["page_0004.png"].issues}
    assert "page_num_skip" in page_two_codes
    assert "page_num_backwards" in page_three_codes
    assert "missing_page_num_between_numbered_pages" in page_four_codes


def test_validate_document_issues_detects_other_pages_between_same_type_and_entity_variants() -> None:
    doc_summary = validate_document_issues(
        [
            ("page_0001.png", PageState(meta={"type": "notes", "entity_name": "Acme Holdings"}, facts=[])),
            ("page_0002.png", PageState(meta={"type": "other", "entity_name": "Acme Holdinqs"}, facts=[])),
            ("page_0003.png", PageState(meta={"type": "notes", "entity_name": "Acme Holdings"}, facts=[])),
        ]
    )
    page_two_codes = {issue.code for issue in doc_summary.page_summaries["page_0002.png"].issues}
    assert "other_page_between_same_type_pages" in page_two_codes
    assert "entity_name_variant" in page_two_codes


def test_aggregate_document_issues_counts_pages_and_totals() -> None:
    page_a = validate_page_issues("page_a.png", PageState(meta={"type": "other"}, facts=[_fact(value="")]))
    page_b = validate_page_issues(
        "page_b.png",
        PageState(meta={"type": "other"}, facts=[_fact(value="10", currency="USD"), _fact(value="11", currency="ILS")]),
    )
    doc_summary = aggregate_document_issues({"page_a.png": page_a, "page_b.png": page_b})
    assert doc_summary.reg_flag_count == 1
    assert doc_summary.warning_count == 2
    assert doc_summary.pages_with_reg_flags == 1
    assert doc_summary.pages_with_warnings == 1
