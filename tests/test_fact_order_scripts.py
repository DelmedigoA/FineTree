from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / name


def test_check_fact_reading_order_script_fails_on_violation(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    payload = {
        "document_meta": {"language": "he"},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other"},
                "facts": [
                    {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "left", "refference": "", "path": []},
                    {"bbox": {"x": 100, "y": 10, "w": 10, "h": 10}, "value": "right", "refference": "", "path": []},
                ],
            }
        ],
    }
    (ann_dir / "doc.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path("check_fact_reading_order.py")), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "reading-order violations" in (proc.stdout + proc.stderr)


def test_fix_and_restore_scripts_roundtrip_with_backup(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    src_path = ann_dir / "doc.json"
    payload = {
        "document_meta": {"language": "he"},
        "pages": [
            {
                "image": "page_0001.png",
                "meta": {"type": "other"},
                "facts": [
                    {"bbox": {"x": 10, "y": 10, "w": 10, "h": 10}, "value": "left", "refference": "", "path": []},
                    {"bbox": {"x": 100, "y": 10, "w": 10, "h": 10}, "value": "right", "refference": "", "path": []},
                ],
            }
        ],
    }
    src_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fix_proc = subprocess.run(
        [sys.executable, str(_script_path("fix_fact_reading_order.py")), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fix_proc.returncode == 0

    fixed = json.loads(src_path.read_text(encoding="utf-8"))
    values = [fact["value"] for fact in fixed["pages"][0]["facts"]]
    assert values == ["right", "left"]

    backups = sorted((tmp_path / "data" / "annotations" / "_backup" / "doc").glob("*.json"))
    assert backups
    backup_path = backups[0]
    manifest = tmp_path / "data" / "annotations" / "_backup" / "manifest.jsonl"
    assert manifest.is_file()
    manifest_lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert manifest_lines
    manifest_row = json.loads(manifest_lines[0])
    assert "timestamp" in manifest_row
    assert "original_path" in manifest_row
    assert "backup_path" in manifest_row

    restore_proc = subprocess.run(
        [
            sys.executable,
            str(_script_path("restore_annotation_backup.py")),
            "--root",
            str(tmp_path),
            "--source",
            str(src_path),
            "--backup",
            str(backup_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert restore_proc.returncode == 0
    restored = json.loads(src_path.read_text(encoding="utf-8"))
    restored_values = [fact["value"] for fact in restored["pages"][0]["facts"]]
    assert restored_values == ["left", "right"]
