from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.fact_ordering import (
    assert_fact_order,
    canonical_fact_order_indices,
    compact_document_meta,
    resolve_reading_direction,
    validate_fact_order,
)


def test_resolve_reading_direction_prefers_explicit_direction() -> None:
    info = resolve_reading_direction(
        {"language": "en", "reading_direction": "rtl"},
        payload={"pages": []},
        default_direction="ltr",
    )
    assert info["direction"] == "rtl"
    assert info["source"] == "document_meta.reading_direction"
    assert info["uncertain"] is False


def test_resolve_reading_direction_uses_language_when_direction_missing() -> None:
    info = resolve_reading_direction({"language": "en"}, payload={"pages": []}, default_direction="rtl")
    assert info["direction"] == "ltr"
    assert info["source"] == "document_meta.language"


def test_resolve_reading_direction_falls_back_with_uncertain_signal() -> None:
    info = resolve_reading_direction({}, payload={"pages": [{"meta": {}, "facts": []}]}, default_direction="rtl")
    assert info["direction"] == "rtl"
    assert info["source"] == "default"
    assert info["uncertain"] is True


def test_canonical_fact_order_indices_respects_hebrew_rtl_order() -> None:
    facts = [
        {"bbox": {"x": 10, "y": 100, "w": 10, "h": 10}, "value": "left"},
        {"bbox": {"x": 100, "y": 100, "w": 10, "h": 10}, "value": "right"},
    ]
    idxs = canonical_fact_order_indices(facts, direction="rtl")
    assert idxs == [1, 0]


def test_canonical_fact_order_indices_respects_english_ltr_order() -> None:
    facts = [
        {"bbox": {"x": 10, "y": 100, "w": 10, "h": 10}, "value": "left"},
        {"bbox": {"x": 100, "y": 100, "w": 10, "h": 10}, "value": "right"},
    ]
    idxs = canonical_fact_order_indices(facts, direction="ltr")
    assert idxs == [0, 1]


def test_validate_fact_order_reports_mismatch() -> None:
    facts = [
        {"bbox": {"x": 10, "y": 100, "w": 10, "h": 10}, "value": "left"},
        {"bbox": {"x": 100, "y": 100, "w": 10, "h": 10}, "value": "right"},
    ]
    report = validate_fact_order(facts, direction="rtl")
    assert report["ok"] is False
    assert report["violations"]


def test_canonical_fact_order_groups_rows_with_tolerance() -> None:
    facts = [
        {"bbox": {"x": 90, "y": 10, "w": 10, "h": 20}, "value": "top-right"},
        {"bbox": {"x": 10, "y": 12, "w": 10, "h": 20}, "value": "top-left"},
        {"bbox": {"x": 90, "y": 100, "w": 10, "h": 20}, "value": "bottom-right"},
        {"bbox": {"x": 10, "y": 102, "w": 10, "h": 20}, "value": "bottom-left"},
    ]
    idxs_rtl = canonical_fact_order_indices(
        facts,
        direction="rtl",
        row_tolerance_ratio=0.35,
        row_tolerance_min_px=6.0,
    )
    idxs_ltr = canonical_fact_order_indices(
        facts,
        direction="ltr",
        row_tolerance_ratio=0.35,
        row_tolerance_min_px=6.0,
    )
    assert idxs_rtl == [0, 1, 2, 3]
    assert idxs_ltr == [1, 0, 3, 2]


def test_canonical_fact_order_is_stable_on_equal_positions() -> None:
    facts = [
        {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "a"},
        {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "b"},
        {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "c"},
    ]
    idxs = canonical_fact_order_indices(facts, direction="ltr")
    assert idxs == [0, 1, 2]


def test_compact_document_meta_drops_invalid_values() -> None:
    out = compact_document_meta({"language": "fr", "reading_direction": "up"})
    assert out == {}


def test_assert_fact_order_detects_issue(tmp_path: Path) -> None:
    ann = tmp_path / "data" / "annotations"
    ann.mkdir(parents=True)
    payload = {
        "document_meta": {"language": "he"},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other"},
                "facts": [
                    {"bbox": {"x": 10, "y": 100, "w": 10, "h": 10}, "value": "left", "refference": "", "path": []},
                    {"bbox": {"x": 100, "y": 100, "w": 10, "h": 10}, "value": "right", "refference": "", "path": []},
                ],
            }
        ],
    }
    (ann / "doc1.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    try:
        assert_fact_order(tmp_path, fail_on_issues=True)
    except RuntimeError as exc:
        assert "reading-order violations" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ordering RuntimeError")
