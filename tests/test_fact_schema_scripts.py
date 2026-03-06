from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / name


def test_check_fact_schema_format_script_fails_on_violation(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other"},
                "facts": [
                    {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "abc", "refference": "", "path": []},
                ],
            }
        ],
    }
    (ann_dir / "doc.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path("check_fact_schema_format.py")), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "schema/format violations" in (proc.stdout + proc.stderr)


def test_fix_fact_schema_format_script_rewrites_with_backup(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    src_path = ann_dir / "doc.json"
    payload = {
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other"},
                "facts": [
                    {
                        "bbox": {"x": 10, "y": 10, "w": 10, "h": 10},
                        "value": "-123",
                        "note": "free text",
                        "is_beur": "",
                        "beur_num": "9",
                        "refference": "2ה׳",
                        "date": "31.12.2024",
                        "path": [],
                    }
                ],
            }
        ],
    }
    src_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fix_proc = subprocess.run(
        [sys.executable, str(_script_path("fix_fact_schema_format.py")), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fix_proc.returncode == 0

    fixed = json.loads(src_path.read_text(encoding="utf-8"))
    fact = fixed["pages"][0]["facts"][0]
    assert fact["value"] == "(123)"
    assert fact["ref_comment"] == "free text"
    assert fact["note_flag"] is False
    assert fact["note_num"] == 9
    assert fact["ref_note"] == "2ה׳"
    assert fact["date"] == "2024-12-31"

    manifest = tmp_path / "data" / "annotations" / "_backup" / "manifest.jsonl"
    assert manifest.is_file()
    line = manifest.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row["reason"] == "fact_schema_format_fix"
