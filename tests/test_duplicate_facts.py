from __future__ import annotations

import json
from pathlib import Path

from finetree_annotator.finetune.duplicate_facts import assert_no_duplicate_facts, duplicate_facts_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_duplicate_facts_report_detects_exact_duplicates(tmp_path: Path) -> None:
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "facts": [
                    {
                        "bbox": [10, 20, 30, 40],
                        "value": "100",
                        "refference": "",
                        "date": "31.12.2024",
                        "path": ["a", "b"],
                    },
                    {
                        "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                        "value": "100",
                        "refference": "",
                        "date": "31.12.2024",
                        "path": ["a", "b"],
                    },
                ],
            }
        ]
    }
    _write_json(tmp_path / "data" / "annotations" / "sample.json", payload)

    report = duplicate_facts_report(tmp_path)
    assert report["files_scanned"] == 1
    assert report["pages_scanned"] == 1
    assert report["facts_scanned"] == 2
    assert report["duplicate_groups"] == 1
    assert report["duplicate_rows"] == 1
    assert len(report["findings"]) == 1
    assert report["findings"][0]["indexes"] == [0, 1]


def test_assert_no_duplicate_facts_can_raise_or_allow(tmp_path: Path) -> None:
    payload = {
        "facts": [
            {"bbox": [1, 2, 3, 4], "value": "x", "refference": "", "path": []},
            {"bbox": [1, 2, 3, 4], "value": "x", "refference": "", "path": []},
        ]
    }
    _write_json(tmp_path / "data" / "annotations" / "single_page.json", payload)

    try:
        assert_no_duplicate_facts(tmp_path)
    except RuntimeError as exc:
        assert "Exact duplicate facts detected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError for duplicate facts")

    allowed = assert_no_duplicate_facts(tmp_path, fail_on_duplicates=False)
    assert allowed["duplicate_rows"] == 1
