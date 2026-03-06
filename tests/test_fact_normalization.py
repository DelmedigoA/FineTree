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
    assert normalized["value"] == "1234"
    assert normalized["comment"] == "*free text"
    assert normalized["is_note"] is False
    assert normalized["note"] == "19"
    assert normalized["note_reference"] == "2ה׳"


def test_normalize_date_accepts_year_and_converts_dmy() -> None:
    year_value, year_warnings = normalize_date("2022")
    dmy_value, dmy_warnings = normalize_date("11.2.2021")
    assert year_value == "2022"
    assert year_warnings == []
    assert dmy_value == "2021-02-11"
    assert dmy_warnings == []


def test_normalize_value_allows_percent_and_normalizes_negative_non_percent() -> None:
    percent_value, percent_warnings = normalize_value(" 12.5% ")
    negative_value, negative_warnings = normalize_value("-123.45")
    assert percent_value == "12.5%"
    assert percent_warnings == []
    assert negative_value == "(123.45)"
    assert negative_warnings == []


def test_normalize_value_dash_and_noncanonical_report_warnings() -> None:
    hyphen_value, hyphen_warnings = normalize_value("-")
    dash_value, dash_warnings = normalize_value("—")
    bad_value, bad_warnings = normalize_value("abc")
    assert hyphen_value == "-"
    assert hyphen_warnings == []
    assert dash_value == ""
    assert "placeholder_value" in dash_warnings
    assert bad_value == "abc"
    assert "noncanonical_value" in bad_warnings


def test_normalize_value_handles_marker_prefixed_numeric_and_currency_dash() -> None:
    marker_value, marker_warnings = normalize_value("*62,565")
    currency_dash_value, currency_dash_warnings = normalize_value("$           -")
    assert marker_value == "62565"
    assert marker_warnings == []
    assert currency_dash_value == "-"
    assert currency_dash_warnings == []


def test_normalize_fact_payload_moves_range_value_to_note_reference() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "7-15",
            "note_reference": "",
            "value_type": "amount",
            "path": [],
        }
    )
    assert normalized["value"] == ""
    assert normalized["note_reference"] == "7-15"
    assert warnings == []


def test_normalize_fact_payload_keeps_range_value_for_percent_type() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "7-15",
            "note_reference": "",
            "value_type": "%",
            "path": [],
        }
    )
    assert normalized["value"] == "7-15"
    assert normalized["note_reference"] is None
    assert normalized["value_type"] == "%"
    assert warnings == []


def test_normalize_fact_payload_coerces_empty_note_reference_to_null() -> None:
    normalized, warnings = normalize_fact_payload(
        {
            "value": "10",
            "note_reference": "",
            "path": [],
        }
    )
    assert normalized["note_reference"] is None
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
    assert normalized["pages"][0]["facts"][0]["comment"] == "legacy free text"
    assert normalized["pages"][0]["facts"][0]["note"] == "7"
    assert normalized["pages"][0]["facts"][0]["is_note"] is False
    assert normalized["pages"][0]["facts"][0]["date"] == "2024-12-31"
    assert normalized["pages"][0]["facts"][0]["value"] == "(123)"
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
