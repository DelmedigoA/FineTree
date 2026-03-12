from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.fact_normalization import (
    assert_fact_format,
    normalize_annotation_payload,
    normalize_date,
    normalize_fact_payload,
    normalize_value,
)


def test_normalize_fact_payload_maps_legacy_keys_to_canonical() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "1,234",
            "note": "*free text",
            "is_beur": "",
            "beur_num": "19",
            "refference": "2ה׳",
            "path": ["assets", "cash"],
        }
    )
    assert warnings == []
    assert normalized["value"] == "1,234"
    assert normalized["comment_ref"] == "*free text"
    assert normalized["note_flag"] is False
    assert normalized["note_num"] == 19
    assert normalized["note_ref"] == "2ה׳"


def test_normalize_date_accepts_year_year_month_and_converts_dmy() -> None:
    year_value, year_warnings = normalize_date("2022")
    year_month_value, year_month_warnings = normalize_date("2024-09")
    dmy_value, dmy_warnings = normalize_date("11.2.2021")
    assert year_value == "2022"
    assert year_warnings == []
    assert year_month_value == "2024-09"
    assert year_month_warnings == []
    assert dmy_value == "2021-02-11"
    assert dmy_warnings == []


def test_normalize_value_preserves_raw_nonempty_text() -> None:
    percent_value, percent_warnings = normalize_value(" 12.5% ")
    negative_value, negative_warnings = normalize_value("-123.45")
    assert percent_value == "12.5%"
    assert percent_warnings == []
    assert negative_value == "-123.45"
    assert negative_warnings == []


def test_normalize_value_dash_and_placeholder_report_warnings() -> None:
    hyphen_value, hyphen_warnings = normalize_value("-")
    dash_value, dash_warnings = normalize_value("—")
    empty_value, empty_warnings = normalize_value("  ")
    bad_value, bad_warnings = normalize_value("abc")
    assert hyphen_value == "-"
    assert hyphen_warnings == []
    assert dash_value == "-"
    assert "placeholder_value" in dash_warnings
    assert empty_value == "-"
    assert "empty_value" in empty_warnings
    assert bad_value == "abc"
    assert bad_warnings == []


def test_normalize_value_handles_marker_prefixed_numeric_and_currency_dash() -> None:
    marker_value, marker_warnings = normalize_value("*62,565")
    currency_dash_value, currency_dash_warnings = normalize_value("$           -")
    assert marker_value == "*62,565"
    assert marker_warnings == []
    assert currency_dash_value == "$           -"
    assert currency_dash_warnings == []


def test_normalize_fact_payload_keeps_range_value_in_value_field() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "7-15",
            "ref_note": "",
            "value_type": "amount",
            "path": [],
        }
    )
    assert normalized["value"] == "7-15"
    assert normalized["note_ref"] is None
    assert warnings == []


def test_normalize_fact_payload_keeps_range_value_for_percent_type() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "7-15",
            "ref_note": "",
            "value_type": "%",
            "path": [],
        }
    )
    assert normalized["value"] == "7-15"
    assert normalized["note_ref"] is None
    assert normalized["value_type"] == "percent"
    assert warnings == []


def test_normalize_fact_payload_preserves_equation() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "100",
            "fact_num": "7",
            "equation": "80 + 20",
            "fact_equation": "f1 + f3",
            "path": [],
        }
    )
    assert normalized["fact_num"] == 7
    assert normalized["equations"] == [{"equation": "80 + 20", "fact_equation": "f1 + f3"}]
    assert warnings == []


def test_normalize_fact_payload_preserves_multiple_equations_and_keeps_active_first() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "120",
            "equation": "100 + 20",
            "fact_equation": "f1 + f2",
            "equations": [
                {
                    "equation": "90 + 30",
                    "fact_equation": "f4 + f5",
                },
                {
                    "equation": "100 + 20",
                    "fact_equation": "f1 + f2",
                },
            ],
            "path": [],
        }
    )
    assert warnings == []
    assert isinstance(normalized["equations"], list)
    assert normalized["equations"][0]["equation"] == "100 + 20"
    assert normalized["equations"][0]["fact_equation"] == "f1 + f2"
    assert normalized["equations"][1]["equation"] == "90 + 30"


def test_normalize_fact_payload_ignores_deprecated_balance_type_and_sets_deterministic_natural_sign() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "(200)",
            "balance_type": "debit",
            "natural_sign": "positive",
            "path": [],
        }
    )
    assert "balance_type" not in normalized
    assert normalized["natural_sign"] == "negative"
    assert warnings == []


def test_normalize_fact_payload_ignores_deprecated_aggregation_role() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "aggregation_role": "subtractive",
            "path": [],
        }
    )
    assert "aggregation_role" not in normalized
    assert warnings == []


def test_normalize_fact_payload_accepts_row_role() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "row_role": "total",
            "path": [],
        }
    )
    assert normalized["row_role"] == "total"
    assert warnings == []


def test_normalize_fact_payload_does_not_infer_aggregation_role_from_path() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "209255",
            "path": ["רכוש קבוע", "בניכוי - פחת שנצבר"],
        }
    )
    assert "aggregation_role" not in normalized
    assert warnings == []


def test_normalize_fact_payload_infers_total_row_role_from_subtotal_path() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "60713",
            "path": ["רכוש קבוע", 'סה"כ רכוש קבוע'],
        }
    )
    assert normalized["row_role"] == "total"
    assert warnings == []


def test_normalize_fact_payload_defaults_numeric_fact_without_aggregation_role() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "269968",
            "path": ["רכוש קבוע", "עלות"],
        }
    )
    assert normalized["row_role"] == "detail"
    assert "aggregation_role" not in normalized
    assert warnings == []


def test_normalize_fact_payload_maps_legacy_total_aggregation_role_to_row_role_total() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "60713",
            "aggregation_role": "total",
            "path": ["רכוש קבוע", 'סה"כ רכוש קבוע'],
        }
    )
    assert normalized["row_role"] == "total"
    assert "aggregation_role" not in normalized
    assert warnings == []


def test_normalize_fact_payload_converts_legacy_equation_children_to_fact_equation() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "120",
            "row_role": "total",
            "equation_children": [{"fact_num": "1", "operator": "+"}, {"fact_num": 2, "operator": "-"}],
            "path": [],
        }
    )
    assert normalized["equations"] is None
    assert "equation_children" not in normalized
    assert warnings == []


def test_normalize_fact_payload_sets_natural_sign_null_for_dash() -> None:
    normalized, warnings = normalize_fact_payload({"value": "-", "path": []})
    assert normalized["natural_sign"] is None
    assert warnings == []


def test_normalize_fact_payload_accepts_recurrent_duration_type() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "duration_type": "recurrent",
            "path": [],
        }
    )
    assert normalized["duration_type"] == "recurrent"
    assert warnings == []


def test_normalize_fact_payload_maps_recurring_duration_type_to_recurrent() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "duration_type": "recurring",
            "path": [],
        }
    )
    assert normalized["duration_type"] == "recurrent"
    assert warnings == []


def test_normalize_fact_payload_keeps_explicit_null_period_range_without_date_inference() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "date": "2024",
            "period_type": None,
            "period_start": None,
            "period_end": None,
            "duration_type": None,
            "path": [],
        }
    )
    assert normalized["date"] == "2024"
    assert normalized["period_type"] is None
    assert normalized["period_start"] is None
    assert normalized["period_end"] is None
    assert warnings == []


def test_normalize_fact_payload_coerces_empty_note_reference_to_null() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "note_reference": "",
            "path": [],
        }
    )
    assert normalized["note_ref"] is None
    assert warnings == []


def test_normalize_annotation_payload_reports_issues() -> None:
    payload = {
        "pages": [
            {
                "image": "p1.png",
                "meta": {"type": "other"},
                "facts": [
                    {
                        "bbox": [1, 2, 3, 4],
                        "value": "-123",
                        "note": "legacy free text",
                        "is_beur": None,
                        "beur_num": "7",
                        "refference": "ref",
                        "date": "31.12.2024",
                        "path": [],
                    }
                ],
            }
        ]
    }
    normalized, findings = normalize_annotation_payload(payload)
    assert normalized["pages"][0]["facts"][0]["comment_ref"] == "legacy free text"
    assert normalized["pages"][0]["facts"][0]["note_num"] == 7
    assert normalized["pages"][0]["facts"][0]["note_flag"] is False
    assert normalized["pages"][0]["facts"][0]["date"] == "2024-12-31"
    assert normalized["pages"][0]["facts"][0]["value"] == "-123"
    assert findings
    assert "legacy_keys" in findings[0]["issue_codes"]


def test_assert_fact_format_raises_on_issues(tmp_path: Path) -> None:
    ann = tmp_path / "data" / "annotations"
    ann.mkdir(parents=True)
    payload = {
        "pages": [
            {
                "image": "p1.png",
                "meta": {"type": "other"},
                "facts": [
                    {"bbox": [1, 2, 3, 4], "value": "A", "refference": "", "path": []},
                ],
            }
        ]
    }
    (ann / "doc.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        assert_fact_format(tmp_path, fail_on_issues=True)
    except RuntimeError as exc:
        assert "schema/format violations" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected schema/format RuntimeError")
