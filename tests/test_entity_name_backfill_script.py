from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / name


def test_backfill_missing_entity_name_uses_first_existing_value(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    src = ann_dir / "doc.json"
    payload = {
        "pages": [
            {"image": "page_0001.png", "meta": {"entity_name": "ACME", "type": "other"}, "facts": []},
            {"image": "page_0002.png", "meta": {"entity_name": None, "type": "other"}, "facts": []},
            {"image": "page_0003.png", "meta": {"entity_name": "", "type": "other"}, "facts": []},
            {"image": "page_0004.png", "meta": {"entity_name": "KEEP", "type": "other"}, "facts": []},
        ]
    }
    src.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path("backfill_missing_entity_name.py")), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0

    fixed = json.loads(src.read_text(encoding="utf-8"))
    pages = fixed["pages"]
    assert pages[0]["meta"]["entity_name"] == "ACME"
    assert pages[1]["meta"]["entity_name"] == "ACME"
    assert pages[2]["meta"]["entity_name"] == "ACME"
    assert pages[3]["meta"]["entity_name"] == "KEEP"


def test_backfill_missing_entity_name_dry_run_does_not_write(tmp_path: Path) -> None:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    src = ann_dir / "doc.json"
    payload = {
        "pages": [
            {"image": "page_0001.png", "meta": {"entity_name": "ACME", "type": "other"}, "facts": []},
            {"image": "page_0002.png", "meta": {"entity_name": None, "type": "other"}, "facts": []},
        ]
    }
    original_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    src.write_text(original_text, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path("backfill_missing_entity_name.py")), "--root", str(tmp_path), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert src.read_text(encoding="utf-8") == original_text
